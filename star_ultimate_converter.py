import os
import re
import json
import time
import glob
import math
import inspect

import torch
import folder_paths
import safetensors
import safetensors.torch
import comfy.utils

from collections import Counter, OrderedDict


try:
    import comfy_kitchen as ck
    from comfy_kitchen.registry import registry as ck_registry
    from comfy_kitchen.tensor import (
        TensorCoreConvRotW4A4Layout,
        TensorCoreMXFP8Layout,
        TensorCoreNVFP4Layout,
        TensorWiseINT8Layout,
    )
    KITCHEN_AVAILABLE = True
except ImportError:
    ck = None
    ck_registry = None
    TensorCoreConvRotW4A4Layout = None
    TensorCoreMXFP8Layout = None
    TensorCoreNVFP4Layout = None
    TensorWiseINT8Layout = None
    KITCHEN_AVAILABLE = False
    print("⚠️ [Star Ultimate Model Converter] comfy-kitchen not found.")


W4A8_LAYOUT = None

try:
    from comfy_kitchen.tensor import AsymW4A8Int8Layout as W4A8_LAYOUT
except ImportError:
    try:
        from comfy_kitchen.tensor.w4a8_int8 import AsymW4A8Int8Layout as W4A8_LAYOUT
    except ImportError:
        W4A8_LAYOUT = None

W4A8_AVAILABLE = W4A8_LAYOUT is not None


NODE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_JSON = os.path.join(NODE_DIR, "models.json")

EXTENDED_METADATA_KEYS = ["config", "license", "encrypted_wandb_properties"]
AIO_MODEL_PREFIX = "model.diffusion_model."


TARGET_FORMATS = [
    "nvfp4",
    "fp8",
    "mxfp8",
    "int8",
    "int8_convrot",
    "int4_convrot",
    "int8_convrot_pruned",
    "int4_convrot_pruned",
    "w4a8_convrot",
    "w4a8_convrot_pruned",
    "minimax_h3_native_mix",
    "svdquant_w4a4",
    "fp16",
    "fp32",
]


MINIMAX_H3_BOUNDARY_BLOCKS = {0, 1, 47, 48, 49}
MINIMAX_H3_NVFP4_GROUPSIZE = 16

CONVROT_GROUPSIZE = 256
INT4_QUANT_GROUPSIZE = 64

SVDQ_OVERSAMPLE = 16
SVDQ_NITER = 2
AWQ_PRESCALE_ALPHA = 0.5

W4A8_GROUP_SIZE = 16
W4A8_CONVROT_GROUPSIZE = 256
W4A8_FORMAT_NAME = "asym_w4a8_int8"

# If your comfy-kitchen W4A8 build rejects symmetric=True, set this to False.
W4A8_SYMMETRIC = True
W4A8_CODEBOOK = True


PRECISION_RE = re.compile(
    r"[-_.]("
    r"fp32|fp16|bf16|mxfp8|"
    r"fp8(?:_e[45]m[23](?:fn)?)?(?:_scaled)?(?:_fast)?|"
    r"int[48](?:_convrot)?(?:_tensorwise)?(?:_pruned)?|"
    r"w4a8(?:_convrot)?(?:_pruned)?|"
    r"nvfp4|svdquant_w4a4"
    r")(?=[-_.]|$)",
    re.IGNORECASE,
)

FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)

DTYPE_NAMES = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
    torch.float8_e4m3fn: "fp8_e4m3fn",
    torch.float8_e5m2: "fp8_e5m2",
    torch.int8: "int8",
}


def detect_input_format(sd, metadata):
    counts = Counter(DTYPE_NAMES.get(v.dtype, str(v.dtype)) for v in sd.values())
    parts = [f"{name} ({n} tensors)" for name, n in counts.most_common()]
    fmt = ", ".join(parts)

    if "scaled_fp8" in sd:
        fmt += " [ComfyUI scaled fp8]"
    elif metadata and "_quantization_metadata" in metadata:
        fmt += " [quantization metadata]"

    return fmt


def format_size(num_bytes):
    return f"{num_bytes / (1024**3):.2f} GB"


def load_model_configs():
    with open(MODELS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def get_profile(configs, model_type):
    default = configs["default"]
    profile = configs["models"].get(model_type, default)

    return (
        profile.get("blacklist", default.get("blacklist", [])),
        profile.get("fp8_layers", default.get("fp8_layers", [])),
        profile.get("preserve_extended_metadata", default.get("preserve_extended_metadata", False)),
        profile.get("force_int8_layers", default.get("force_int8_layers", [])),
        profile.get("awq_prescale_layers", default.get("awq_prescale_layers", [])),
        profile.get("pruned_extra_blacklist", default.get("pruned_extra_blacklist", [])),
        profile.get("int8_convrot_layers", default.get("int8_convrot_layers", [])),
        profile.get("int4_convrot_layers", default.get("int4_convrot_layers", [])),
        profile.get("keep_fp32", default.get("keep_fp32", [])),
        profile.get("keep_fp16", default.get("keep_fp16", [])),
        profile.get("w4a8_blacklist", default.get("w4a8_blacklist", [])),
        profile.get("w4a8_keep_fp32", default.get("w4a8_keep_fp32", [])),
        profile.get("w4a8_keep_fp16", default.get("w4a8_keep_fp16", [])),
    )


def blacklisted_dtype(k: str, keep_fp32, keep_fp16) -> torch.dtype:
    if keep_fp32 and any(name in k for name in keep_fp32):
        return torch.float32

    if keep_fp16 and any(name in k for name in keep_fp16):
        return torch.float16

    return torch.bfloat16


def resolve_input(mode, diffusion_model, checkpoint, text_encoder, custom_path, vae="None"):
    """Resolve the target path based on the selected mode."""

    if mode == "Custom Path":
        custom_path = (custom_path or "").strip().strip('"')
        if not custom_path:
            raise ValueError("Mode is 'Custom Path' but no custom path was provided.")

        src = os.path.abspath(os.path.expanduser(custom_path))

        if os.path.isdir(src):
            files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
            if not files:
                raise ValueError(f"No .safetensors files found in: {src}")
            return files, os.path.dirname(src), os.path.basename(src)

        if os.path.isfile(src):
            return [src], os.path.dirname(src), os.path.splitext(os.path.basename(src))[0]

        raise ValueError(f"Path not found: {src}")

    elif mode == "Diffusion Model":
        if not diffusion_model or diffusion_model == "None":
            raise ValueError("No Diffusion Model selected. Please choose a model from the dropdown.")

        path = folder_paths.get_full_path("diffusion_models", diffusion_model)
        if not path:
            raise ValueError(f"Diffusion Model not found: {diffusion_model}")

        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    elif mode == "VAE":
        if not vae or vae == "None":
            raise ValueError("No VAE selected. Please choose a model from the dropdown.")

        path = folder_paths.get_full_path("vae", vae)
        if not path:
            raise ValueError(f"VAE not found: {vae}")

        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    elif mode == "Text-Encoder":
        if not text_encoder or text_encoder == "None":
            raise ValueError("No Text-Encoder selected. Please choose a model from the dropdown.")

        path = folder_paths.get_full_path("text_encoders", text_encoder)
        if not path:
            path = folder_paths.get_full_path("clip", text_encoder)

        if not path:
            raise ValueError(f"Text-Encoder not found: {text_encoder}")

        return [path], os.path.dirname(path), os.path.splitext(os.path.basename(path))[0]

    raise ValueError(f"Unknown mode: {mode}")


def diffusion_models_dir():
    paths = folder_paths.get_folder_paths("diffusion_models")

    for p in paths:
        if os.path.basename(os.path.normpath(p)) == "diffusion_models":
            return p

    return paths[0]


def load_aio_model(checkpoint_name):
    """Load an AIO checkpoint and return only its diffusion model state dict."""
    ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint_name)
    if not ckpt_path:
        raise ValueError(f"Checkpoint not found: {checkpoint_name}")

    full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = {
        k[len(AIO_MODEL_PREFIX):]: v
        for k, v in full_sd.items()
        if k.startswith(AIO_MODEL_PREFIX)
    }

    if not sd:
        raise ValueError(
            f"No '{AIO_MODEL_PREFIX}' keys found in {os.path.basename(ckpt_path)}. "
            "Is this an all-in-one checkpoint?"
        )

    return sd, ckpt_path


def pick_mxfp8_backend(device):
    if not KITCHEN_AVAILABLE or TensorCoreMXFP8Layout is None:
        raise RuntimeError("MXFP8 requires comfy-kitchen TensorCoreMXFP8Layout.")

    probe = torch.randn(32, 32, device=device, dtype=torch.float32)

    try:
        TensorCoreMXFP8Layout.quantize(probe)
        return None
    except Exception as e:
        print(f"⚠️ MXFP8 default backend failed ({e}). Trying fallback backends...")

    for backend in ("triton", "eager"):
        try:
            with ck_registry.use_backend(backend):
                TensorCoreMXFP8Layout.quantize(probe)
            print(f"✅ MXFP8: using '{backend}' backend")
            return backend
        except Exception:
            continue

    raise RuntimeError(
        "MXFP8 quantization is not supported by any comfy_kitchen backend in this environment. "
        "Try updating comfy-kitchen and PyTorch."
    )


def compute_awq_prescale(weight: torch.Tensor, alpha: float = AWQ_PRESCALE_ALPHA) -> torch.Tensor:
    w = weight.float()
    channel_mag = w.abs().mean(dim=0).clamp(min=1e-5)
    s = channel_mag.pow(alpha)
    s = (s / s.mean()).clamp(min=1e-4)
    return s


def pack_int4_nibbles(q: torch.Tensor) -> torch.Tensor:
    """Retained for reference only."""
    if q.shape[1] % 2 != 0:
        raise ValueError(f"in_features {q.shape[1]} is odd, cannot pack 2-per-byte cleanly.")

    low = (q[:, 0::2] & 0x0F).to(torch.uint8)
    high = (q[:, 1::2] & 0x0F).to(torch.uint8)
    packed = low | (high << 4)

    return packed


def block_index_from_key(k: str):
    if not k.startswith("blocks."):
        return None

    parts = k.split(".", 2)

    try:
        return int(parts[1])
    except (IndexError, ValueError):
        return None


def svd_lowrank(weight: torch.Tensor, rank: int, oversample: int = SVDQ_OVERSAMPLE, niter: int = SVDQ_NITER):
    w = weight.float()
    min_dim = min(w.shape)
    rank = max(1, min(int(rank), min_dim))
    q = min(rank + max(0, int(oversample)), min_dim)

    if min_dim <= max(q, 32):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r, s_r, vh_r = u[:, :rank], s[:rank], vh[:rank, :]
    else:
        u, s, v = torch.svd_lowrank(w, q=q, niter=max(0, int(niter)))
        u_r, s_r, vh_r = u[:, :rank], s[:rank], v[:, :rank].transpose(-2, -1)

    return (u_r * s_r.unsqueeze(0)).contiguous(), vh_r.contiguous()


def svdquant_split(weight: torch.Tensor, rank: int, groupsize: int, refine_iters: int):
    w = weight.float()
    w_norm = torch.linalg.matrix_norm(w).item()

    if not math.isfinite(w_norm) or w_norm == 0.0:
        return None

    layout = TensorCoreConvRotW4A4Layout

    qw = torch.zeros((), device=w.device, dtype=torch.float32)
    best = None
    best_err = float("inf")

    for _ in range(max(1, refine_iters)):
        target = w - qw
        l1, l2 = svd_lowrank(target, rank, oversample=SVDQ_OVERSAMPLE, niter=SVDQ_NITER)

        l1 = l1.to(torch.bfloat16)
        l2 = l2.to(torch.bfloat16)
        lw = l1.float() @ l2.float()

        residual = (w - lw).to(torch.bfloat16)
        qdata, params = layout.quantize(residual.float().contiguous(), convrot_groupsize=groupsize)
        qw = layout.dequantize(qdata, params).float()

        del qdata, params

        left = w - (lw + qw)
        err = (torch.linalg.matrix_norm(left) / w_norm).item()

        del lw, left

        if not math.isfinite(err):
            break

        if best is not None and err >= best_err - 1e-6:
            break

        best_err, best = err, (residual, l1, l2)

    return best


def build_output_path(out_dir, base_name, target_format):
    stem = PRECISION_RE.sub("", base_name).rstrip("-_.")
    return os.path.join(out_dir, f"{stem}-{target_format}.safetensors")


def load_input(files):
    sd = {}

    for i, fp in enumerate(files):
        if len(files) > 1:
            print(f"📦 Loading shard {i + 1}/{len(files)}: {os.path.basename(fp)}")

        part = comfy.utils.load_torch_file(fp, safe_load=True)

        for k in part:
            if k in sd:
                print(f"⚠️ Duplicate key '{k}' in {os.path.basename(fp)}, overwriting")

        sd.update(part)

    with safetensors.safe_open(files[0], framework="pt") as f:
        orig_meta = f.metadata()

    return sd, orig_meta


def assign_quantized_tensor(new_sd, key, tensor):
    if E8M0_DTYPE is not None and tensor.dtype == E8M0_DTYPE:
        new_sd[key] = tensor.view(torch.uint8).cpu()
    elif tensor.dtype in FP8_DTYPES:
        new_sd[key] = tensor.view(torch.uint8).cpu().view(tensor.dtype)
    else:
        new_sd[key] = tensor.cpu()


def store_quantized_weight(new_sd, original_weight_key, tensors):
    """
    Stores layout tensors returned by comfy_kitchen state_dict_tensors().

    original_weight_key is the original '.weight' key, e.g.:
        blocks.0.attn.out_proj.weight

    suffixes are expected to be things like:
        ""
        "_codebook"
        "_s_channel"
        "_s_rel"
        "_scale"
    """
    for suffix, tensor in tensors.items():
        if not suffix:
            out_key = original_weight_key
        elif suffix.startswith(".") or suffix.startswith("_"):
            out_key = original_weight_key + suffix
        else:
            out_key = f"{original_weight_key}.{suffix}"

        assign_quantized_tensor(new_sd, out_key, tensor)


def quantize_w4a8_convrot(weight_f32: torch.Tensor):
    if not W4A8_AVAILABLE:
        raise RuntimeError(
            "W4A8 ConvRot requires comfy-kitchen AsymW4A8Int8Layout. "
            "Update comfy-kitchen to a build that contains it."
        )

    try:
        sig = inspect.signature(W4A8_LAYOUT.quantize)
        kwargs = {}

        if "group_size" in sig.parameters:
            kwargs["group_size"] = W4A8_GROUP_SIZE
        elif "quant_group_size" in sig.parameters:
            kwargs["quant_group_size"] = W4A8_GROUP_SIZE

        if "convrot_groupsize" in sig.parameters:
            kwargs["convrot_groupsize"] = W4A8_CONVROT_GROUPSIZE

        if "convrot" in sig.parameters:
            kwargs["convrot"] = True

        if "codebook" in sig.parameters:
            kwargs["codebook"] = W4A8_CODEBOOK

        if "symmetric" in sig.parameters:
            kwargs["symmetric"] = W4A8_SYMMETRIC

        return W4A8_LAYOUT.quantize(weight_f32, **kwargs)

    except TypeError:
        # Fallback to the known reference-style direct call.
        return W4A8_LAYOUT.quantize(
            weight_f32,
            group_size=W4A8_GROUP_SIZE,
            convrot_groupsize=W4A8_CONVROT_GROUPSIZE,
            symmetric=W4A8_SYMMETRIC,
            codebook=W4A8_CODEBOOK,
        )


def dequantize_input(sd, metadata):
    """Unquantize fp8/int8 inputs back to bf16 so mixed-precision models re-quantize cleanly."""
    quant_layers = {}

    if metadata and "_quantization_metadata" in metadata:
        quant_layers = json.loads(metadata["_quantization_metadata"]).get("layers", {})

    for k in [k for k in sd if k.endswith(".comfy_quant")]:
        conf = sd.pop(k)
        layer = k[: -len(".comfy_quant")]

        if layer not in quant_layers:
            try:
                quant_layers[layer] = json.loads(bytes(conf.cpu().to(torch.uint8).tolist()))
            except Exception:
                print(f"⚠️ Could not parse embedded quant config for '{layer}', ignoring.")

    for layer, info in quant_layers.items():
        fmt = info.get("format")

        if fmt in ("nvfp4", "mxfp8", "convrot_w4a4", "asym_w4a8_int8"):
            raise ValueError(
                f"Input model contains {fmt} layers ('{layer}'), "
                "which cannot be dequantized losslessly. Use a higher precision source model."
            )

        if info.get("convrot") and fmt not in ("convrot_w4a4", "asym_w4a8_int8"):
            raise ValueError(
                f"Input model contains ConvRot-rotated INT8 layers ('{layer}'). "
                "Use a higher precision source model."
            )

    if "scaled_fp8" in sd:
        sd.pop("scaled_fp8")

        for k in [k for k in sd if k.endswith(".scale_weight")]:
            scale = sd.pop(k)
            wk = k[: -len(".scale_weight")] + ".weight"

            if wk in sd:
                sd[wk] = (sd[wk].to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)

        for k in [k for k in sd if k.endswith(".scale_input")]:
            sd.pop(k)

    for k in list(sd.keys()):
        if k not in sd or not k.endswith(".weight"):
            continue

        v = sd[k]

        if v.dtype in FP8_DTYPES or v.dtype == torch.int8:
            scale = sd.pop(k + "_scale", None)

            if scale is not None:
                sd[k] = (v.to(torch.float32) * scale.to(torch.float32)).to(torch.bfloat16)
            elif v.dtype == torch.int8:
                raise ValueError(f"int8 weight '{k}' has no '{k}_scale' tensor, cannot dequantize.")

    for k, v in sd.items():
        if v.dtype in FP8_DTYPES:
            sd[k] = v.to(torch.bfloat16)

    return sd


class StarUltimateModelConverter:
    @classmethod
    def INPUT_TYPES(s):
        configs = load_model_configs()

        tenc_list = []

        if "text_encoders" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("text_encoders") or [])

        if "clip" in folder_paths.folder_names_and_paths:
            tenc_list.extend(folder_paths.get_filename_list("clip") or [])

        tenc_list = sorted(list(set(tenc_list)))
        tenc_list.insert(0, "None")

        diff_list = folder_paths.get_filename_list("diffusion_models") or []
        diff_list = sorted(diff_list)
        diff_list.insert(0, "None")

        ckpt_list = folder_paths.get_filename_list("checkpoints") or []
        ckpt_list = sorted(ckpt_list)
        ckpt_list.insert(0, "None")

        vae_list = folder_paths.get_filename_list("vae") or []
        vae_list = sorted(vae_list)
        vae_list.insert(0, "None")

        return {
            "required": {
                "mode": (
                    ["Diffusion Model", "Checkpoint", "AIO", "Text-Encoder", "VAE", "Custom Path"],
                    {
                        "default": "Diffusion Model",
                        "tooltip": "Select the source type. 'AIO' processes UNet + CLIP together and ignores VAE. 'VAE' targets the models/vae/ folder.",
                    },
                ),
                "diffusion_model": (diff_list, {"tooltip": "Used if Mode is 'Diffusion Model'."}),
                "checkpoint": (ckpt_list, {"tooltip": "Used if Mode is 'Checkpoint' or 'AIO'."}),
                "text_encoder": (tenc_list, {"tooltip": "Used if Mode is 'Text-Encoder'."}),
                "vae": (vae_list, {"tooltip": "Used if Mode is 'VAE'."}),
                "model_type": (
                    list(configs["models"].keys()),
                    {"tooltip": "Choose the model architecture profile."},
                ),
                "target_format": (TARGET_FORMATS, {"default": "nvfp4"}),
                "device": (["cuda", "cpu"], {"default": "cpu"}),
            },
            "optional": {
                "custom_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                        "placeholder": "Enter path to .safetensors file or folder",
                        "tooltip": "Used only if Mode is set to 'Custom Path'.",
                    },
                ),
                "svdquant_rank": (
                    "INT",
                    {
                        "default": 64,
                        "min": 8,
                        "max": 512,
                        "step": 8,
                        "tooltip": "[svdquant_w4a4 only] Rank of the low-rank bf16 branch pulled out of each weight before quantizing the residual. Higher = better fidelity, larger file.",
                    },
                ),
                "svdquant_refine_iters": (
                    "INT",
                    {
                        "default": 10,
                        "min": 0,
                        "max": 200,
                        "tooltip": "[svdquant_w4a4 only] Refine the low-rank branch against the quantization error, keeping the best split.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "convert"
    CATEGORY = "⭐StarNodes/Model Tools"
    OUTPUT_NODE = True

    def convert(
        self,
        mode,
        diffusion_model,
        checkpoint,
        text_encoder,
        model_type,
        target_format,
        device,
        custom_path="",
        vae="None",
        svdquant_rank=64,
        svdquant_refine_iters=10,
    ):
        configs = load_model_configs()

        (
            blacklist,
            fp8_layers,
            preserve_extended,
            force_int8_layers,
            awq_prescale_layers,
            pruned_extra_blacklist,
            int8_convrot_layers,
            int4_convrot_layers,
            keep_fp32,
            keep_fp16,
            w4a8_blacklist,
            w4a8_keep_fp32,
            w4a8_keep_fp16,
        ) = get_profile(configs, model_type)

        (
            te_blacklist,
            te_fp8_layers,
            _,
            te_force_int8_layers,
            te_awq_prescale_layers,
            te_pruned_extra_blacklist,
            te_int8_convrot_layers,
            te_int4_convrot_layers,
            te_keep_fp32,
            te_keep_fp16,
            te_w4a8_blacklist,
            te_w4a8_keep_fp32,
            te_w4a8_keep_fp16,
        ) = get_profile(configs, "Text-Encoder")

        is_pruned_format = target_format.endswith("_pruned")
        is_w4a8 = target_format in ("w4a8_convrot", "w4a8_convrot_pruned")

        if target_format not in ("fp16", "fp32"):
            if is_w4a8:
                if not W4A8_AVAILABLE:
                    raise ValueError(
                        "W4A8 ConvRot requires comfy-kitchen with AsymW4A8Int8Layout. "
                        "Update comfy-kitchen or use a build that includes the W4A8 layout."
                    )
            elif not KITCHEN_AVAILABLE:
                raise ValueError("comfy-kitchen is required for this target format.")

        start_time = time.time()

        print(f"🚀 [Star Ultimate Model Converter] Mode: {mode} | Profile: {model_type} | Target: {target_format}")

        if target_format == "svdquant_w4a4":
            print(f"🧬 SVDQuant rank: {svdquant_rank}  refine_iters: {svdquant_refine_iters}")

        if mode in ("Checkpoint", "AIO"):
            if not checkpoint or checkpoint == "None":
                raise ValueError(f"Mode is '{mode}' but no checkpoint is selected. Please choose a model from the dropdown.")

            ckpt_path = folder_paths.get_full_path("checkpoints", checkpoint)
            base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
            orig_meta = None

            if ckpt_path.endswith(".safetensors"):
                with safetensors.safe_open(ckpt_path, framework="pt") as f:
                    orig_meta = f.metadata()

            full_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

            if mode == "Checkpoint":
                print(f"✂️ Extracting diffusion model from AIO checkpoint: {checkpoint}")

                sd = {
                    k[len(AIO_MODEL_PREFIX):]: v
                    for k, v in full_sd.items()
                    if k.startswith(AIO_MODEL_PREFIX)
                }

                input_bytes = sum(v.numel() * v.element_size() for v in sd.values())
                output_path = build_output_path(diffusion_models_dir(), base_name, target_format)
                files = [ckpt_path]

                del full_sd

            else:
                print(f"🔄 AIO Mode: Processing entire checkpoint intact: {checkpoint}")

                sd = full_sd
                input_bytes = os.path.getsize(ckpt_path)
                checkpoints_dir = folder_paths.get_folder_paths("checkpoints")[0]
                output_path = build_output_path(checkpoints_dir, f"{base_name}_AIO", target_format)
                files = [ckpt_path]

        else:
            files, out_dir, base_name = resolve_input(
                mode,
                diffusion_model,
                checkpoint,
                text_encoder,
                custom_path,
                vae,
            )

            output_path = build_output_path(out_dir, base_name, target_format)
            input_bytes = sum(os.path.getsize(f) for f in files)
            sd, orig_meta = load_input(files)

        temp_diffusers_meta = {}

        if orig_meta:
            if "format" in orig_meta:
                temp_diffusers_meta["format"] = orig_meta["format"]

            if "modelspec.architecture" in orig_meta:
                temp_diffusers_meta["modelspec.architecture"] = orig_meta["modelspec.architecture"]

            if preserve_extended:
                for key in EXTENDED_METADATA_KEYS:
                    if key in orig_meta:
                        temp_diffusers_meta[key] = orig_meta[key]

        input_format = detect_input_format(sd, orig_meta)
        sd = dequantize_input(sd, orig_meta)

        quant_map = {"format_version": "1.0", "layers": {}}
        new_sd = {}
        counts = Counter()

        pbar = comfy.utils.ProgressBar(len(sd))

        print(f"⚙️ Converting on: {device}")

        mxfp8_backend = pick_mxfp8_backend(device) if target_format == "mxfp8" else None

        if target_format in ("fp16", "fp32"):
            target_dtype = torch.float16 if target_format == "fp16" else torch.float32

            for i, (k, v) in enumerate(sd.items()):
                pbar.update_absolute(i + 1)

                if v.dtype.is_floating_point:
                    new_sd[k] = v.to(target_dtype)
                    counts[target_format] += 1
                else:
                    new_sd[k] = v
                    counts["kept"] += 1

        else:
            for i, (k, v) in enumerate(sd.items()):
                pbar.update_absolute(i + 1)

                if mode == "AIO":
                    if k.startswith(AIO_MODEL_PREFIX):
                        active_blacklist = blacklist
                        active_fp8 = fp8_layers
                        active_force_int8 = force_int8_layers
                        active_awq_prescale = awq_prescale_layers
                        active_pruned_extra = pruned_extra_blacklist
                        active_int8_convrot = int8_convrot_layers
                        active_int4_convrot = int4_convrot_layers
                        active_keep_fp32 = keep_fp32
                        active_keep_fp16 = keep_fp16
                        active_w4a8_blacklist = w4a8_blacklist
                        active_w4a8_keep_fp32 = w4a8_keep_fp32
                        active_w4a8_keep_fp16 = w4a8_keep_fp16

                    elif (
                        k.startswith("cond_stage_model.")
                        or k.startswith("conditioner.")
                        or k.startswith("text_encoders.")
                    ):
                        active_blacklist = te_blacklist
                        active_fp8 = te_fp8_layers
                        active_force_int8 = te_force_int8_layers
                        active_awq_prescale = te_awq_prescale_layers
                        active_pruned_extra = te_pruned_extra_blacklist
                        active_int8_convrot = te_int8_convrot_layers
                        active_int4_convrot = te_int4_convrot_layers
                        active_keep_fp32 = te_keep_fp32
                        active_keep_fp16 = te_keep_fp16
                        active_w4a8_blacklist = te_w4a8_blacklist
                        active_w4a8_keep_fp32 = te_w4a8_keep_fp32
                        active_w4a8_keep_fp16 = te_w4a8_keep_fp16

                    else:
                        if v.dtype.is_floating_point:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (VAE/Misc)"] += 1
                        else:
                            new_sd[k] = v
                            counts["kept (VAE/Misc)"] += 1

                        continue

                else:
                    active_blacklist = blacklist
                    active_fp8 = fp8_layers
                    active_force_int8 = force_int8_layers
                    active_awq_prescale = awq_prescale_layers
                    active_pruned_extra = pruned_extra_blacklist
                    active_int8_convrot = int8_convrot_layers
                    active_int4_convrot = int4_convrot_layers
                    active_keep_fp32 = keep_fp32
                    active_keep_fp16 = keep_fp16
                    active_w4a8_blacklist = w4a8_blacklist
                    active_w4a8_keep_fp32 = w4a8_keep_fp32
                    active_w4a8_keep_fp16 = w4a8_keep_fp16

                if is_w4a8:
                    if active_w4a8_blacklist:
                        active_blacklist = list(active_blacklist) + [
                            x for x in active_w4a8_blacklist if x not in active_blacklist
                        ]

                    if active_w4a8_keep_fp32:
                        active_keep_fp32 = list(active_keep_fp32) + [
                            x for x in active_w4a8_keep_fp32 if x not in active_keep_fp32
                        ]

                    if active_w4a8_keep_fp16:
                        active_keep_fp16 = list(active_keep_fp16) + [
                            x for x in active_w4a8_keep_fp16 if x not in active_keep_fp16
                        ]

                if is_pruned_format and active_pruned_extra:
                    if any(name in k for name in active_pruned_extra) and not any(name in k for name in active_blacklist):
                        active_blacklist = list(active_blacklist) + [n for n in active_pruned_extra if n in k]

                if any(name in k for name in active_blacklist):
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=blacklisted_dtype(k, active_keep_fp32, active_keep_fp16))
                        counts["kept bf16/f16/f32"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1

                    continue

                if v.ndim == 2 and ".weight" in k:
                    base_k_file = k.replace(".weight", "")
                    base_k_meta = base_k_file

                    v_tensor = v.to(device=device, dtype=torch.bfloat16)

                    if (
                        active_force_int8
                        and any(name in k for name in active_force_int8)
                        and target_format not in ("int4_tensorwise", "int4_tensorwise_pruned")
                    ):
                        print(f"🔒 FORCE-INT8: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()
                            qdata, params = TensorWiseINT8Layout.quantize(v_tensor_ready)
                            tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {"format": "int8_tensorwise"}
                            counts["forced_int8"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ Forced INT8 failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if active_int4_convrot and any(name in k for name in active_int4_convrot):
                        print(f"💎 OVERRIDE int4_convrot: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            qdata, params = TensorCoreConvRotW4A4Layout.quantize(
                                v_tensor_ready,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                                quant_group_size=INT4_QUANT_GROUPSIZE,
                            )

                            tensors = TensorCoreConvRotW4A4Layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": "convrot_w4a4",
                                "convrot_groupsize": CONVROT_GROUPSIZE,
                                "quant_group_size": INT4_QUANT_GROUPSIZE,
                            }

                            counts["override_int4_convrot"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ OVERRIDE int4_convrot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if active_int8_convrot and any(name in k for name in active_int8_convrot):
                        print(f"💎 OVERRIDE int8_convrot: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            qdata, params = TensorWiseINT8Layout.quantize(
                                v_tensor_ready,
                                per_channel=True,
                                convrot=True,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                            )

                            tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": "int8_tensorwise",
                                "convrot": True,
                                "convrot_groupsize": CONVROT_GROUPSIZE,
                            }

                            counts["override_int8_convrot"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ OVERRIDE int8_convrot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if target_format == "fp8" or (active_fp8 and any(name in k for name in active_fp8)):
                        print(f"🌸 FP8: {k}")

                        weight_scale = (v_tensor.abs().max() / 448.0).clamp(min=1e-12).float()
                        weight_quantized = ck.quantize_per_tensor_fp8(v_tensor, weight_scale)

                        new_sd[k] = weight_quantized.cpu()
                        new_sd[f"{base_k_file}.weight_scale"] = weight_scale.to(torch.bfloat16).cpu()

                        quant_map["layers"][base_k_meta] = {"format": "float8_e4m3fn"}
                        counts["fp8"] += 1

                        if device == "cuda":
                            del v_tensor

                        continue

                    if target_format == "svdquant_w4a4":
                        print(f"💎 SVDQUANT_W4A4 (rank {svdquant_rank}): {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            split = svdquant_split(
                                v_tensor_ready,
                                svdquant_rank,
                                CONVROT_GROUPSIZE,
                                svdquant_refine_iters,
                            )

                            if split is None:
                                print(
                                    f"  warning: {k} is degenerate (zero or non-finite); "
                                    f"quantizing without a low-rank branch"
                                )
                                residual = v_tensor_ready.to(torch.bfloat16)
                                l1 = l2 = None
                            else:
                                residual, l1, l2 = split

                            layout = TensorCoreConvRotW4A4Layout

                            qdata, params = layout.quantize(
                                residual.float().contiguous(),
                                convrot_groupsize=CONVROT_GROUPSIZE,
                            )

                            tensors = layout.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            layer_conf = {
                                "format": "convrot_w4a4",
                                "convrot_groupsize": CONVROT_GROUPSIZE,
                                "svdquant": l1 is not None,
                                "svdquant_rank": svdquant_rank,
                            }

                            if l1 is not None:
                                new_sd[f"{base_k_file}.svdq_l1"] = l1.cpu()
                                new_sd[f"{base_k_file}.svdq_l2"] = l2.cpu()

                            quant_map["layers"][base_k_meta] = layer_conf
                            counts["svdquant_w4a4"] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ SVDQuant failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    if target_format == "minimax_h3_native_mix":
                        blk_idx = block_index_from_key(k)

                        if "attn.out_proj" in k:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (out_proj)"] += 1
                            continue

                        if blk_idx in MINIMAX_H3_BOUNDARY_BLOCKS:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16 (boundary block)"] += 1
                            continue

                        if "attn.qkv_proj" in k:
                            print(f"💎 NATIVE_MIX int8_convrot: {k}")

                            try:
                                v_tensor_ready = v_tensor.float().contiguous()

                                qdata, params = TensorWiseINT8Layout.quantize(
                                    v_tensor_ready,
                                    per_channel=True,
                                    convrot=True,
                                    convrot_groupsize=CONVROT_GROUPSIZE,
                                )

                                tensors = TensorWiseINT8Layout.state_dict_tensors(qdata, params)

                                store_quantized_weight(new_sd, k, tensors)

                                quant_map["layers"][base_k_meta] = {
                                    "format": "int8_tensorwise",
                                    "convrot": True,
                                    "convrot_groupsize": CONVROT_GROUPSIZE,
                                }

                                counts["native_mix_qkv_int8convrot"] += 1

                                if device == "cuda":
                                    del v_tensor, v_tensor_ready

                            except Exception as e:
                                print(f"⚠️ NATIVE_MIX qkv_proj failed for {k}: {e}")

                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1

                                if device == "cuda":
                                    del v_tensor

                            continue

                        if "mlp.fc1" in k or "mlp.fc2" in k:
                            print(f"💎 NATIVE_MIX nvfp4: {k}")

                            try:
                                v_tensor_ready = v_tensor.float().contiguous()

                                qdata, params = TensorCoreNVFP4Layout.quantize(v_tensor_ready)
                                tensors = TensorCoreNVFP4Layout.state_dict_tensors(qdata, params)

                                store_quantized_weight(new_sd, k, tensors)

                                quant_map["layers"][base_k_meta] = {
                                    "format": "nvfp4",
                                    "group_size": MINIMAX_H3_NVFP4_GROUPSIZE,
                                }

                                counts["native_mix_mlp_nvfp4"] += 1

                                if device == "cuda":
                                    del v_tensor, v_tensor_ready

                            except Exception as e:
                                print(f"⚠️ NATIVE_MIX mlp failed for {k}: {e}")

                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["kept bf16"] += 1

                                if device == "cuda":
                                    del v_tensor

                            continue

                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16 (native_mix other)"] += 1
                        continue

                    if is_w4a8:
                        print(f"💎 W4A8_CONVROT: {k}")

                        try:
                            v_tensor_ready = v_tensor.float().contiguous()

                            qdata, params = quantize_w4a8_convrot(v_tensor_ready)
                            tensors = W4A8_LAYOUT.state_dict_tensors(qdata, params)

                            store_quantized_weight(new_sd, k, tensors)

                            quant_map["layers"][base_k_meta] = {
                                "format": W4A8_FORMAT_NAME,
                                "group_size": W4A8_GROUP_SIZE,
                                "convrot": True,
                                "convrot_groupsize": W4A8_CONVROT_GROUPSIZE,
                            }

                            counts[target_format] += 1

                            if device == "cuda":
                                del v_tensor, v_tensor_ready

                        except Exception as e:
                            print(f"⚠️ W4A8 ConvRot failed for {k}: {e}")

                            if v.dtype.is_floating_point:
                                new_sd[k] = v.to(dtype=torch.bfloat16)
                                counts["w4a8_failed_bf16"] += 1
                            else:
                                new_sd[k] = v
                                counts["kept"] += 1

                            if device == "cuda":
                                del v_tensor

                        continue

                    int8_convrot = target_format in ("int8_convrot", "int8_convrot_pruned")
                    int4_convrot = target_format in ("int4_convrot", "int4_convrot_pruned")

                    if target_format in ("int8", "int8_convrot", "int8_convrot_pruned"):
                        layout = TensorWiseINT8Layout
                        fmt_name = "int8_tensorwise"
                    elif target_format in ("int4_convrot", "int4_convrot_pruned"):
                        layout = TensorCoreConvRotW4A4Layout
                        fmt_name = "convrot_w4a4"
                    elif target_format == "mxfp8":
                        layout = TensorCoreMXFP8Layout
                        fmt_name = "mxfp8"
                    else:
                        layout = TensorCoreNVFP4Layout
                        fmt_name = "nvfp4"

                    print(f"💎 {target_format.upper()}: {k}")

                    try:
                        v_tensor_ready = v_tensor.float().contiguous()
                        pre_quant_scale = None

                        if (
                            target_format == "nvfp4"
                            and active_awq_prescale
                            and any(name in k for name in active_awq_prescale)
                        ):
                            pre_quant_scale = compute_awq_prescale(v_tensor_ready)
                            v_tensor_ready = v_tensor_ready * pre_quant_scale.unsqueeze(0)

                        if int8_convrot:
                            qdata, params = layout.quantize(
                                v_tensor_ready,
                                per_channel=True,
                                convrot=True,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                            )
                        elif int4_convrot:
                            qdata, params = layout.quantize(
                                v_tensor_ready,
                                convrot_groupsize=CONVROT_GROUPSIZE,
                                quant_group_size=INT4_QUANT_GROUPSIZE,
                            )
                        elif target_format == "mxfp8" and mxfp8_backend is not None:
                            with ck_registry.use_backend(mxfp8_backend):
                                qdata, params = layout.quantize(v_tensor_ready)
                        else:
                            qdata, params = layout.quantize(v_tensor_ready)

                        tensors = layout.state_dict_tensors(qdata, params)

                        store_quantized_weight(new_sd, k, tensors)

                        if pre_quant_scale is not None:
                            new_sd[f"{base_k_file}.pre_quant_scale"] = pre_quant_scale.to(torch.bfloat16).cpu()

                        layer_conf = {"format": fmt_name}

                        if int8_convrot:
                            layer_conf["convrot"] = True
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                        elif int4_convrot:
                            layer_conf["convrot_groupsize"] = CONVROT_GROUPSIZE
                            layer_conf["quant_group_size"] = INT4_QUANT_GROUPSIZE

                        if pre_quant_scale is not None:
                            layer_conf["awq_prescale"] = True

                        quant_map["layers"][base_k_meta] = layer_conf
                        counts[target_format] += 1

                    except Exception as e:
                        print(f"⚠️ Quantization failed for {k}: {e}")

                        if v.dtype.is_floating_point:
                            new_sd[k] = v.to(dtype=torch.bfloat16)
                            counts["kept bf16"] += 1
                        else:
                            new_sd[k] = v
                            counts["kept"] += 1

                    if device == "cuda":
                        del v_tensor

                else:
                    if v.dtype.is_floating_point:
                        new_sd[k] = v.to(dtype=torch.bfloat16)
                        counts["kept bf16"] += 1
                    else:
                        new_sd[k] = v
                        counts["kept"] += 1

        final_metadata = OrderedDict()

        if quant_map["layers"]:
            final_metadata["_quantization_metadata"] = json.dumps(quant_map)

        final_metadata["converted_by"] = "Star Ultimate Model Converter"

        if target_format == "svdquant_w4a4":
            final_metadata["svdquant_rank"] = str(svdquant_rank)
            final_metadata["svdquant_refine_iters"] = str(svdquant_refine_iters)

        for k, v in temp_diffusers_meta.items():
            final_metadata[k] = v

        print(f"💾 Saving | Type: {model_type} | Path: {output_path}")

        safetensors.torch.save_file(new_sd, output_path, metadata=final_metadata)

        output_bytes = os.path.getsize(output_path)
        duration = time.time() - start_time
        reduction = (1 - output_bytes / input_bytes) * 100 if input_bytes else 0

        print(f"✅ Done. Final size: {format_size(output_bytes)}")

        if mode == "AIO":
            input_desc = f"Full AIO Checkpoint {os.path.basename(files[0])}"
        elif mode == "Checkpoint":
            input_desc = f"diffusion model from AIO checkpoint {os.path.basename(files[0])}"
        elif len(files) > 1:
            input_desc = f"{len(files)} files from {os.path.basename(os.path.dirname(files[0]))}"
        else:
            input_desc = os.path.basename(files[0])

        layers_desc = ", ".join(f"{n} {name}" for name, n in counts.most_common())

        status = "\n".join(
            [
                f"✅ Success ({model_type} → {target_format})",
                f"Input: {input_desc}",
                f"Original format: {input_format}",
                f"Original size: {format_size(input_bytes)}",
                f"New size: {format_size(output_bytes)} ({reduction:.1f}% smaller)",
                f"Layers: {layers_desc}",
                f"Device: {device} | Time: {duration:.1f}s",
                f"Saved to: {output_path}",
            ]
        )

        return (status,)


NODE_CLASS_MAPPINGS = {
    "StarUltimateModelConverter": StarUltimateModelConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StarUltimateModelConverter": "⭐ Star Ultimate Model Converter",
}