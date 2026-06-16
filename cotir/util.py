import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from .flux2.sampling import get_schedule as get_schedule_flux2
from .flux.sampling import get_schedule as get_schedule_flux, unpack as unpack_flux
from typing import Any
from pathlib import Path
import os

import numpy as np
from PIL import Image

from .flux.util import configs as FLUX_MODEL_CONFIGS, load_ae as load_ae_flux, load_clip, load_t5
from .flux2.util import FLUX2_MODEL_INFO, load_ae as load_ae_flux2, load_text_encoder

# =========================
# Common tensor/image utils
# =========================

# =========================
# Transformers torch.load safety patch (for qalign metric)
# =========================
# Some environments ship torch < 2.6, while `transformers` now enforces
# a safety gate for `torch.load`. `pyiqa`'s `qalign` metric depends on
# a legacy model loading path and may fail unless we bypass the gate.
_TRANSFORMERS_TORCH_LOAD_SAFETY_PATCHED = False


def patch_transformers_torch_load_safety_once() -> bool:
    """
    Patch transformers' internal `check_torch_load_is_safe` to bypass the
    torch>=2.6 version restriction.

    This is intended for local/trusted evaluation only.
    """
    global _TRANSFORMERS_TORCH_LOAD_SAFETY_PATCHED
    if _TRANSFORMERS_TORCH_LOAD_SAFETY_PATCHED:
        return True
    try:
        import transformers.utils.import_utils as iu
        import transformers.modeling_utils as mu

        iu.check_torch_load_is_safe = lambda: None
        mu.check_torch_load_is_safe = lambda: None
        _TRANSFORMERS_TORCH_LOAD_SAFETY_PATCHED = True
        print(
            "[warn] Enabled transformers torch.load safety patch for qalign. "
            "Use only with locally trusted weights.",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"[warn] Failed to enable torch.load safety patch: {exc}", flush=True)
        return False

def check_tensor_consistency(data_dict, expected_device, expected_dtype, func_name=""):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            if v.device != expected_device:
                data_dict[k] = v.to(expected_device)
            if v.dtype.is_floating_point and v.dtype != expected_dtype:
                data_dict[k] = v.to(expected_dtype)


def resize_long_edge(image: Image.Image, target: int | None) -> Image.Image:
    if target is None:
        return image
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= target:
        return image
    scale = target / float(long_edge)
    new_w = max(16, int(round(width * scale)))
    new_h = max(16, int(round(height * scale)))
    return image.resize((new_w, new_h), Image.BICUBIC)


def _pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array / 127.5 - 1.0).permute(2, 0, 1).contiguous()


def _tensor_to_uint8_hwc(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) * 127.5
    return tensor.to(torch.uint8).permute(1, 2, 0).numpy()


def load_image_as_tensor(path: Path, max_long_edge: int | None = None) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = resize_long_edge(img, max_long_edge)
    return _pil_to_normalized_tensor(img)


def load_pil_image_as_tensor(
    image: Image.Image,
    max_long_edge: int | None = None,
    keep_multiple_of_16: bool = False,
) -> torch.Tensor:
    image = image.convert("RGB")
    image = resize_long_edge(image, max_long_edge)
    if keep_multiple_of_16:
        width, height = image.size
        new_w = max(16, (width // 16) * 16)
        new_h = max(16, (height // 16) * 16)
        if (new_w, new_h) != (width, height):
            image = image.resize((new_w, new_h), Image.BICUBIC)
    return _pil_to_normalized_tensor(image)


def tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(_tensor_to_uint8_hwc(tensor))

# =========================
# Inference config/backends
# =========================


def _get_expected_base_filename(base_model_name: str) -> str:
    if base_model_name.startswith("flux.2"):
        if base_model_name not in FLUX2_MODEL_INFO:
            raise KeyError(f"Unsupported base model: {base_model_name}. Keys={list(FLUX2_MODEL_INFO.keys())}")
        return FLUX2_MODEL_INFO[base_model_name]["filename"]
    if base_model_name not in FLUX_MODEL_CONFIGS:
        raise KeyError(f"Unsupported base model: {base_model_name}. Keys={list(FLUX_MODEL_CONFIGS.keys())}")
    return FLUX_MODEL_CONFIGS[base_model_name].repo_flow


def _resolve_flux2_model_dir(cfg, base_ckpt: Path | None) -> Path:
    if base_ckpt is not None:
        return base_ckpt.parent.parent if base_ckpt.parent.name == "transformer" else base_ckpt.parent
    return Path(str(cfg.inference.base_model_path)).expanduser().resolve()


def _resolve_kontext_component_paths(kontext_root: Path) -> dict[str, Path]:
    return {
        "t5_dir": kontext_root / "text_encoder_2",
        "clip_dir": kontext_root / "text_encoder",
        "t5_tokenizer_dir": kontext_root / "tokenizer_2",
        "clip_tokenizer_dir": kontext_root / "tokenizer",
        "ae_path": kontext_root / "ae.safetensors",
    }


def _validate_kontext_component_paths(paths: dict[str, Path]) -> None:
    checks = [
        ("Local T5 directory not found", "t5_dir"),
        ("Local CLIP directory not found", "clip_dir"),
        ("Local T5 tokenizer directory not found", "t5_tokenizer_dir"),
        ("Local CLIP tokenizer directory not found", "clip_tokenizer_dir"),
        ("Local AE safetensors not found", "ae_path"),
    ]
    for message, key in checks:
        if not paths[key].exists():
            raise FileNotFoundError(f"{message}: {paths[key]}")


def resolve_inference_paths(cfg) -> tuple[Path, Path, Path]:
    if not hasattr(cfg, "inference"):
        raise ValueError("Config missing `inference` section.")
    if not hasattr(cfg.inference, "base_model_path"):
        raise ValueError("Config missing `inference.base_model_path`.")
    if not hasattr(cfg.inference, "cot_adapter_path"):
        raise ValueError("Config missing `inference.cot_adapter_path`.")
    if not hasattr(cfg.inference, "lora_path"):
        raise ValueError("Config missing `inference.lora_path`.")
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "base_model_name"):
        raise ValueError("Config missing `model.base_model_name`.")

    base_model_path = Path(str(cfg.inference.base_model_path)).expanduser().resolve()
    cot_ckpt = Path(str(cfg.inference.cot_adapter_path)).expanduser().resolve()
    lora_ckpt = Path(str(cfg.inference.lora_path)).expanduser().resolve()

    base_model_name = str(cfg.model.base_model_name).lower()
    expected_filename = _get_expected_base_filename(base_model_name)

    if base_model_path.is_file():
        base_ckpt = base_model_path
    else:
        base_ckpt = base_model_path / expected_filename
        if not base_ckpt.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_ckpt}")

    return base_ckpt, cot_ckpt, lora_ckpt


def read_infer_prompt(prompt_file: str | None, lq_path: Path, samples_root: Path) -> str:
    if prompt_file:
        prompt_path = Path(prompt_file).expanduser().resolve()
    else:
        prompt_path = samples_root / "prompts" / f"{lq_path.stem}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return prompt


def load_infer_backends_by_model(
    cfg,
    device: torch.device,
    base_ckpt: Path | None = None,
) -> tuple[Any, Any, Any]:
    base_model_name = str(cfg.model.base_model_name).lower()
    if base_model_name.startswith("flux.2"):
        flux2_model_dir = _resolve_flux2_model_dir(cfg, base_ckpt)
        os.environ["FLUX2_MODEL_DIR"] = str(flux2_model_dir)
        t5 = load_text_encoder(cfg.model.base_model_name, device=device).requires_grad_(False).eval()
        clip = None
        ae = load_ae_flux2(cfg.model.base_model_name, device=device).requires_grad_(False).eval()
        return t5, clip, ae

    kontext_root = Path(str(cfg.inference.base_model_path)).expanduser().resolve()
    kontext_paths = _resolve_kontext_component_paths(kontext_root)
    _validate_kontext_component_paths(kontext_paths)
    os.environ["FLUX_T5_PATH"] = str(kontext_paths["t5_dir"])
    os.environ["FLUX_CLIP_PATH"] = str(kontext_paths["clip_dir"])
    os.environ["FLUX_T5_TOKENIZER_PATH"] = str(kontext_paths["t5_tokenizer_dir"])
    os.environ["FLUX_CLIP_TOKENIZER_PATH"] = str(kontext_paths["clip_tokenizer_dir"])
    os.environ["FLUX_AE"] = str(kontext_paths["ae_path"])
    t5 = load_t5(device=device).requires_grad_(False).eval()
    clip = load_clip(device=device).requires_grad_(False).eval()
    ae = load_ae_flux(cfg.model.base_model_name, device=device).requires_grad_(False).eval()
    return t5, clip, ae

# =========================
# Inference data/sampling
# =========================


def _prepare_infer_inputs(
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompt: str,
    ae: Any,
    t5: Any,
    device: torch.device,
) -> dict[str, Any]:
    bs = hq_img.shape[0]
    pos_dtype = torch.float32
    with torch.no_grad():
        x_1 = ae.encode(hq_img.to(device))
        x_c = ae.encode(lq_img.to(device))
        _, _, h_lat, w_lat = x_1.shape
        x_1 = rearrange(x_1, "b c h w -> b (h w) c")
        x_c = rearrange(x_c, "b c h w -> b (h w) c")

        t_target = torch.tensor([0.0], device=device, dtype=pos_dtype)
        t_cond = torch.tensor([10.0], device=device, dtype=pos_dtype)
        h_idx = torch.arange(h_lat, device=device, dtype=pos_dtype)
        w_idx = torch.arange(w_lat, device=device, dtype=pos_dtype)
        l_idx = torch.tensor([0.0], device=device, dtype=pos_dtype)

        img_ids = torch.cartesian_prod(t_target, h_idx, w_idx, l_idx).unsqueeze(0).expand(bs, -1, -1)
        img_cond_ids = torch.cartesian_prod(t_cond, h_idx, w_idx, l_idx).unsqueeze(0).expand(bs, -1, -1)

        txt = t5([prompt] * bs)
        seq_len = txt.shape[1]
        txt_ids = torch.zeros(bs, seq_len, 4, device=device, dtype=pos_dtype)
        txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=pos_dtype)
        vec = torch.zeros(bs, 1, device=device, dtype=pos_dtype)

    return {
        "x_1": x_1,
        "x_c": x_c,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "img_ids": img_ids,
        "img_cond_ids": img_cond_ids,
        "txt": txt.to(device),
        "txt_ids": txt_ids,
        "vec": vec,
    }


def _prepare_infer_inputs_kontext(
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompt: str,
    ae: Any,
    t5: Any,
    clip: Any,
    device: torch.device,
) -> dict[str, Any]:
    bs = hq_img.shape[0]
    pos_dtype = torch.float32
    with torch.no_grad():
        x_1 = ae.encode(hq_img.to(device))
        x_c = ae.encode(lq_img.to(device))
        _, _, h, w = x_1.shape
        x_1 = rearrange(x_1, "b c (hh ph) (ww pw) -> b (hh ww) (c ph pw)", ph=2, pw=2)
        x_c = rearrange(x_c, "b c (hh ph) (ww pw) -> b (hh ww) (c ph pw)", ph=2, pw=2)

        img_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=pos_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device, dtype=pos_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device, dtype=pos_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        img_cond_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=pos_dtype)
        img_cond_ids[..., 0] = 1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(h // 2, device=device, dtype=pos_dtype)[:, None]
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(w // 2, device=device, dtype=pos_dtype)[None, :]
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

        prompts = [prompt] * bs
        txt = t5(prompts)
        txt_ids = torch.zeros(bs, txt.shape[1], 3, device=device, dtype=pos_dtype)
        if clip is None:
            vec = torch.zeros(bs, 1, device=device, dtype=pos_dtype)
        else:
            vec = clip(prompts).to(device)

    return {
        "x_1": x_1,
        "x_c": x_c,
        "h_lat": h // 2,
        "w_lat": w // 2,
        "img_ids": img_ids,
        "img_cond_ids": img_cond_ids,
        "txt": txt.to(device),
        "txt_ids": txt_ids,
        "vec": vec,
        "packed_latent": True,
    }


def prepare_infer_inputs_by_model(
    base_model_name: str,
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompt: str,
    ae: Any,
    t5: Any,
    device: torch.device,
    clip: Any = None,
) -> dict[str, Any]:
    if str(base_model_name).lower().startswith("flux.2"):
        data = _prepare_infer_inputs(hq_img=hq_img, lq_img=lq_img, prompt=prompt, ae=ae, t5=t5, device=device)
        data["packed_latent"] = False
        return data
    return _prepare_infer_inputs_kontext(
        hq_img=hq_img, lq_img=lq_img, prompt=prompt, ae=ae, t5=t5, clip=clip, device=device
    )


def _prepare_infer_inputs_flux2_batch(
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompts: list[str],
    ae: Any,
    t5: Any,
    device: torch.device,
) -> dict[str, Any]:
    bs = hq_img.shape[0]
    pos_dtype = torch.float32
    with torch.no_grad():
        x_1 = ae.encode(hq_img.to(device))
        x_c = ae.encode(lq_img.to(device))
        _, _, h_lat, w_lat = x_1.shape
        x_1 = rearrange(x_1, "b c h w -> b (h w) c")
        x_c = rearrange(x_c, "b c h w -> b (h w) c")

        t_target = torch.tensor([0.0], device=device, dtype=pos_dtype)
        t_cond = torch.tensor([10.0], device=device, dtype=pos_dtype)
        h_idx = torch.arange(h_lat, device=device, dtype=pos_dtype)
        w_idx = torch.arange(w_lat, device=device, dtype=pos_dtype)
        l_idx = torch.tensor([0.0], device=device, dtype=pos_dtype)
        img_ids = torch.cartesian_prod(t_target, h_idx, w_idx, l_idx).unsqueeze(0).expand(bs, -1, -1)
        img_cond_ids = torch.cartesian_prod(t_cond, h_idx, w_idx, l_idx).unsqueeze(0).expand(bs, -1, -1)

        txt = t5(prompts)
        seq_len = txt.shape[1]
        txt_ids = torch.zeros(bs, seq_len, 4, device=device, dtype=pos_dtype)
        txt_ids[..., 3] = torch.arange(seq_len, device=device, dtype=pos_dtype)
        vec = torch.zeros(bs, 1, device=device, dtype=pos_dtype)

    return {
        "x_1": x_1,
        "x_c": x_c,
        "h_lat": h_lat,
        "w_lat": w_lat,
        "img_ids": img_ids,
        "img_cond_ids": img_cond_ids,
        "txt": txt.to(device),
        "txt_ids": txt_ids,
        "vec": vec,
        "packed_latent": False,
    }


def _prepare_infer_inputs_kontext_batch(
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompts: list[str],
    ae: Any,
    t5: Any,
    clip: Any,
    device: torch.device,
) -> dict[str, Any]:
    bs = hq_img.shape[0]
    pos_dtype = torch.float32
    with torch.no_grad():
        x_1 = ae.encode(hq_img.to(device))
        x_c = ae.encode(lq_img.to(device))
        _, _, h, w = x_1.shape
        x_1 = rearrange(x_1, "b c (hh ph) (ww pw) -> b (hh ww) (c ph pw)", ph=2, pw=2)
        x_c = rearrange(x_c, "b c (hh ph) (ww pw) -> b (hh ww) (c ph pw)", ph=2, pw=2)

        img_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=pos_dtype)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2, device=device, dtype=pos_dtype)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2, device=device, dtype=pos_dtype)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

        img_cond_ids = torch.zeros(h // 2, w // 2, 3, device=device, dtype=pos_dtype)
        img_cond_ids[..., 0] = 1
        img_cond_ids[..., 1] = img_cond_ids[..., 1] + torch.arange(h // 2, device=device, dtype=pos_dtype)[:, None]
        img_cond_ids[..., 2] = img_cond_ids[..., 2] + torch.arange(w // 2, device=device, dtype=pos_dtype)[None, :]
        img_cond_ids = repeat(img_cond_ids, "h w c -> b (h w) c", b=bs)

        txt = t5(prompts)
        txt_ids = torch.zeros(bs, txt.shape[1], 3, device=device, dtype=pos_dtype)
        vec = clip(prompts).to(device) if clip is not None else torch.zeros(bs, 1, device=device, dtype=pos_dtype)

    return {
        "x_1": x_1,
        "x_c": x_c,
        "h_lat": h // 2,
        "w_lat": w // 2,
        "img_ids": img_ids,
        "img_cond_ids": img_cond_ids,
        "txt": txt.to(device),
        "txt_ids": txt_ids,
        "vec": vec,
        "packed_latent": True,
    }


def prepare_infer_inputs_by_model_with_prompts(
    base_model_name: str,
    hq_img: torch.Tensor,
    lq_img: torch.Tensor,
    prompts: list[str],
    ae: Any,
    t5: Any,
    device: torch.device,
    clip: Any = None,
) -> dict[str, Any]:
    if str(base_model_name).lower().startswith("flux.2"):
        return _prepare_infer_inputs_flux2_batch(
            hq_img=hq_img,
            lq_img=lq_img,
            prompts=prompts,
            ae=ae,
            t5=t5,
            device=device,
        )
    return _prepare_infer_inputs_kontext_batch(
        hq_img=hq_img,
        lq_img=lq_img,
        prompts=prompts,
        ae=ae,
        t5=t5,
        clip=clip,
        device=device,
    )


def sample_and_decode_by_model(
    model: Any,
    ae: Any,
    infer_data: dict[str, Any],
    hq_img: torch.Tensor,
    base_model_name: str,
    num_steps: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    img = _sample_image_tokens(
        model=model,
        infer_data=infer_data,
        base_model_name=base_model_name,
        num_steps=num_steps,
        device=device,
        weight_dtype=weight_dtype,
        seed=seed,
    )

    if infer_data.get("packed_latent", False):
        out_img_latent = unpack_flux(img.to(torch.float32), hq_img.shape[2], hq_img.shape[3])
    else:
        out_img_latent = rearrange(
            img.to(torch.float32),
            "b (h w) c -> b c h w",
            h=infer_data["h_lat"],
            w=infer_data["w_lat"],
        )
    return ae.decode(out_img_latent)


def _sample_image_tokens(
    model: Any,
    infer_data: dict[str, Any],
    base_model_name: str,
    num_steps: int,
    device: torch.device,
    weight_dtype: torch.dtype,
    seed: int | None = None,
) -> torch.Tensor:
    is_flux2 = str(base_model_name).lower().startswith("flux.2")
    x_1 = infer_data["x_1"]
    x_c = infer_data["x_c"]
    img_ids = infer_data["img_ids"]
    img_cond_ids = infer_data["img_cond_ids"]
    txt = infer_data["txt"]
    txt_ids = infer_data["txt_ids"]
    vec = infer_data["vec"]

    if is_flux2:
        timesteps = get_schedule_flux2(int(num_steps), x_1.shape[1])
    else:
        timesteps = get_schedule_flux(int(num_steps), x_1.shape[1], shift=True)

    if seed is None:
        img = torch.randn_like(x_1, device=device)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        img = torch.randn(x_1.shape, generator=gen, device=device, dtype=x_1.dtype)

    guidance_vec = torch.full((img.shape[0],), 4, device=img.device, dtype=img.dtype)
    img_input_ids = torch.cat((img_ids, img_cond_ids), dim=1)

    with torch.no_grad():
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            img_input = torch.cat((img, x_c), dim=1)
            model_outputs = model(
                img=img_input.to(weight_dtype),
                img_ids=img_input_ids.to(weight_dtype),
                txt=txt.to(weight_dtype),
                txt_ids=txt_ids.to(weight_dtype),
                y=vec.to(weight_dtype),
                timesteps=t_vec.to(weight_dtype),
                guidance=guidance_vec.to(weight_dtype),
            )
            pred = model_outputs[0][:, : img.shape[1]]
            img = img + (t_prev - t_curr) * pred
    return img


def _tokens_to_latent_grid(
    tokens: torch.Tensor,
    infer_data: dict[str, Any],
    hq_img: torch.Tensor,
) -> torch.Tensor:
    if infer_data.get("packed_latent", False):
        return unpack_flux(tokens.to(torch.float32), hq_img.shape[2], hq_img.shape[3])
    return rearrange(
        tokens.to(torch.float32),
        "b (h w) c -> b c h w",
        h=infer_data["h_lat"],
        w=infer_data["w_lat"],
    )


def _resolve_num_steps_from_args(args, default_steps: int = 20) -> int:
    inference = getattr(args, "inference", None)
    if inference is not None and hasattr(inference, "num_steps"):
        return int(inference.num_steps)
    training = getattr(args, "training", None)
    if training is not None and hasattr(training, "num_steps"):
        return int(training.num_steps)
    return int(default_steps)


def _resolve_base_model_name_from_args(args, default_name: str = "flux.2") -> str:
    model_cfg = getattr(args, "model", None)
    if model_cfg is not None and hasattr(model_cfg, "base_model_name"):
        return str(model_cfg.base_model_name).lower()
    return default_name

# =========================
# Train/val loop
# =========================

def prepare_kontext_train_data(
    hq_img,
    lq_img,
    prompts_gen,
    prompts_spe,
    answer_s,
    answer_d,
    answer_p,
    ae,
    t5,
    clip,
    device,
    weight_dtype,
    use_gen_prob,
    base_model_name: str = "flux.2-klein-4b",
):
    """
    Prepare Kontext training data (training version based on prepare_kontext function)
    
    Args:
        hq_img: High quality image tensor [B, C, H, W]
        lq_img: Low quality/conditioning image tensor [B, C, H, W]
        prompts: Text prompt list
        answer: Answer text list
        ae: Autoencoder
        t5: T5 text encoder
        clip: CLIP text encoder
        device: Device
        weight_dtype: Weight data type
        
    Returns:
        dict: Dictionary containing all training data
            - x_1: High quality image latent [B, C, H, W]
            - x_c: Conditioning image latent [B, C, H, W]
            - x_c_packed: Packed conditioning image latent [B, H*W/4, C*4]
            - img_ids: Target image IDs [B, H*W/4, 3]
            - img_cond_ids: Conditioning image IDs [B, H*W/4, 3]
            - txt: Encoded text [B, seq_len, dim]
            - txt_ids: Text IDs [B, seq_len, 3]
            - vec: CLIP text vector [B, dim]
            - txt_gt: Encoded answer text [B, seq_len, dim] (if provided)
            - bs: Batch size
            - h: Latent height
            - w: Latent width
    """
    bs = hq_img.shape[0]
    
    with torch.no_grad():
        # Mix prompts from generated/specific sets based on a Bernoulli mask sampled per element
        use_gen_prob = float(use_gen_prob) if use_gen_prob is not None else 0.0
        use_gen_prob = max(0.0, min(1.0, use_gen_prob))
        prompts_gen_list = list(prompts_gen)
        prompts_spe_list = list(prompts_spe)
        
        if len(prompts_gen_list) != bs or len(prompts_spe_list) != bs:
            raise ValueError(
                f"Mismatch between batch size ({bs}) and prompt list lengths: "
                f"gen={len(prompts_gen_list)}, spe={len(prompts_spe_list)}"
            )

        if use_gen_prob <= 0.0:
            mixed_prompts = prompts_spe_list
        elif use_gen_prob >= 1.0:
            mixed_prompts = prompts_gen_list
        else:
            bern_mask = torch.rand(bs) < use_gen_prob
            mask_list = bern_mask.tolist()
            mixed_prompts = [
                prompts_gen_list[idx] if mask_list[idx] else prompts_spe_list[idx]
                for idx in range(bs)
            ]
        prompts = mixed_prompts
        data = prepare_infer_inputs_by_model_with_prompts(
            base_model_name=base_model_name,
            hq_img=hq_img,
            lq_img=lq_img,
            prompts=prompts,
            ae=ae,
            t5=t5,
            clip=clip,
            device=device,
        )
        
        # Encode answer text (if needed)
        answers_concat = list(answer_s) + list(answer_d) + list(answer_p)
        encoded = t5(answers_concat)  # shape: [3*B, seq, dim]

        txt_gt_s, txt_gt_d, txt_gt_p = torch.split(encoded, bs, dim=0)
    
    result = {
        'x_1': data['x_1'],
        'x_c': data['x_c'],
        'h_lat': data['h_lat'],
        'w_lat': data['w_lat'],
        'packed_latent': data.get('packed_latent', False),
        'prompts': prompts,
        'img_ids': data['img_ids'].to(device),
        'img_cond_ids': data['img_cond_ids'].to(device),
        'txt': data['txt'].to(device),
        'txt_ids': data['txt_ids'].to(device),
        'vec': data['vec'].to(device),
        'txt_gt_s': txt_gt_s.to(device),
        'txt_gt_d': txt_gt_d.to(device),
        'txt_gt_p': txt_gt_p.to(device),
    }
    
    # Check tensor consistency
    check_tensor_consistency(result, device, weight_dtype, func_name="prepare_kontext_train_data")
    
    return result


def train_one_step(
    batch, model,
    lambda_s, lambda_d, lambda_p,
    delta_s, delta_d, delta_p,
    use_gen_prob,
    t5, clip, ae,
    device,
    weight_dtype,
    base_model_name: str = "flux.2-klein-4b",
    ):

    hq_img, lq_img, prompts_gen, prompts_spe, answer_s, answer_d, answer_p = batch
    bs = hq_img.shape[0]
    
    # Prepare training data
    data = prepare_kontext_train_data(
        hq_img, lq_img, prompts_gen, prompts_spe, answer_s, answer_d, answer_p,
        ae, t5, clip, device, weight_dtype, use_gen_prob, base_model_name=base_model_name
    )
    
    # Unpack data
    x_1 = data['x_1']
    x_c = data['x_c']
    img_ids = data['img_ids']
    img_cond_ids = data['img_cond_ids']
    txt_ids = data['txt_ids']
    txt = data['txt']
    txt_gt_s = data['txt_gt_s']
    txt_gt_d = data['txt_gt_d']
    txt_gt_p = data['txt_gt_p']

    vec = data['vec']
    
    # Sample t and r for CoTIR training
    x_0 = torch.randn_like(x_1).to(device)
    t = torch.sigmoid(torch.randn((bs,), device=device))
    t_ = rearrange(t, "b -> b 1 1").detach().clone()
    x_t = (1 - t_) * x_1 + t_ * x_0

    guidance_vec = torch.full((x_t.shape[0],), 4, device=x_t.device, dtype=x_t.dtype)

    img_input = torch.cat((x_t, x_c), dim=1)
    img_input_ids = torch.cat((img_ids, img_cond_ids), dim=1)
    
    pred_img = None

    pred_img, pred_txt_s, pred_txt_d, pred_txt_p = model(
        img=img_input.to(weight_dtype),
        img_ids=img_input_ids.to(weight_dtype),
        txt=txt.to(weight_dtype),
        txt_ids=txt_ids.to(weight_dtype),
        y=vec.to(weight_dtype),
        timesteps=t.to(weight_dtype),
        guidance=guidance_vec.to(weight_dtype),
    )
    pred_img = pred_img[:, : x_0.shape[1]]

    img_loss = F.mse_loss(pred_img.float(), (x_0 - x_1).float(), reduction="mean")

    pred_txt_s_n = F.normalize(pred_txt_s.float(), p=2, dim=-1, eps=1e-6)
    pred_txt_d_n = F.normalize(pred_txt_d.float(), p=2, dim=-1, eps=1e-6)
    pred_txt_p_n = F.normalize(pred_txt_p.float(), p=2, dim=-1, eps=1e-6)
    txt_gt_s_n = F.normalize(txt_gt_s.float(), p=2, dim=-1, eps=1e-6)
    txt_gt_d_n = F.normalize(txt_gt_d.float(), p=2, dim=-1, eps=1e-6)
    txt_gt_p_n = F.normalize(txt_gt_p.float(), p=2, dim=-1, eps=1e-6)

    txt_loss_s = F.mse_loss(pred_txt_s_n, txt_gt_s_n, reduction="mean")
    txt_loss_d = F.mse_loss(pred_txt_d_n, txt_gt_d_n, reduction="mean")
    txt_loss_p = F.mse_loss(pred_txt_p_n, txt_gt_p_n, reduction="mean")
    c_s = txt_loss_s - delta_s
    c_d = txt_loss_d - delta_d
    c_p = txt_loss_p - delta_p
    loss = img_loss + lambda_s.detach() * c_s + lambda_d.detach() * c_d + lambda_p.detach() * c_p

    return loss, img_loss, txt_loss_s, txt_loss_d, txt_loss_p, c_s, c_d, c_p


def val_one_step(
    model, batch,
    args,
    t5, clip, ae,
    device,
    weight_dtype,
    use_gen_prob=1,
    ):

    hq_img, lq_img, prompts_gen, prompts_spe, answer_s, answer_d, answer_p = batch
    bs = hq_img.shape[0]
    
    # Generate shared noise seed for fair comparison
    # Use a fixed seed for reproducibility (same noise for both gen and spe)
    shared_noise_seed = 42
    
    def run_inference(use_gen_prompt, noise_seed=None):
        """Run inference with either gen or spe prompts"""
        with torch.no_grad():
            base_model_name = _resolve_base_model_name_from_args(args)
            prompts = list(prompts_gen) if use_gen_prompt else list(prompts_spe)
            data = prepare_infer_inputs_by_model_with_prompts(
                base_model_name=base_model_name,
                hq_img=hq_img,
                lq_img=lq_img,
                prompts=prompts,
                ae=ae,
                t5=t5,
                clip=clip,
                device=device,
            )

            x_1 = data['x_1']
            x_c = data['x_c']
            img = _sample_image_tokens(
                model=model,
                infer_data=data,
                base_model_name=base_model_name,
                num_steps=_resolve_num_steps_from_args(args),
                device=device,
                weight_dtype=weight_dtype,
                seed=noise_seed,
            )

            out_img_latent = _tokens_to_latent_grid(img, data, hq_img)
            rec_lq_latent = _tokens_to_latent_grid(x_c, data, hq_img)
            rec_hq_latent = _tokens_to_latent_grid(x_1, data, hq_img)

            out_img = ae.decode(out_img_latent)
            rec_lq = ae.decode(rec_lq_latent)
            rec_hq = ae.decode(rec_hq_latent)
        
        return out_img, rec_lq, rec_hq
    
    # Run inference with gen prompts (using shared noise seed)
    out_img_gen, rec_lq_gen, rec_hq_gen = run_inference(use_gen_prompt=True, noise_seed=shared_noise_seed)
    
    # Run inference with spe prompts (using same shared noise seed for fair comparison)
    out_img_spe, rec_lq_spe, rec_hq_spe = run_inference(use_gen_prompt=False, noise_seed=shared_noise_seed)
    
    # Convert prompts to lists if they are not already
    prompts_gen_list = list(prompts_gen) if not isinstance(prompts_gen, list) else prompts_gen
    prompts_spe_list = list(prompts_spe) if not isinstance(prompts_spe, list) else prompts_spe
    
    answers_out = [
        " | ".join(
            [segment for segment in (
                f"scene: {scene}" if scene else None,
                f"degradation: {deg}" if deg else None,
                f"plan: {plan}" if plan else None,
            ) if segment is not None]
        )
        for scene, deg, plan in zip(answer_s, answer_d, answer_p)
    ]

    results = {
        'out_img_gen': out_img_gen,
        'out_img_spe': out_img_spe,
        'rec_lq': rec_lq_gen,  # Same for both, use gen version
        'rec_hq': rec_hq_gen,  # Same for both, use gen version
        'lq_img': lq_img,
        'hq_img': hq_img,
        'prompts_gen': prompts_gen_list,
        'prompts_spe': prompts_spe_list,
        'answer': answers_out,
    }
    
    return results


# =========================
# Visualization/checkpoint
# =========================

def _strip_ddp_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def _is_lora_key(key: str) -> bool:
    return (
        ".lora_" in key
        or ".lora_A." in key
        or ".lora_B." in key
        or key.startswith("lora.")
        or ".lora." in key
    )


def _is_cot_adapter_key(key: str) -> bool:
    return key.startswith("cot_adapter.") or ".cot_adapter." in key


def _split_state_dict_for_save(
    raw_sd: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    base_sd: dict[str, torch.Tensor] = {}
    lora_sd: dict[str, torch.Tensor] = {}
    cot_sd: dict[str, torch.Tensor] = {}
    for k, v in raw_sd.items():
        if _is_cot_adapter_key(k):
            cot_sd[k] = v
        elif _is_lora_key(k):
            lora_sd[k] = v
        else:
            base_sd[k] = v
    return base_sd, lora_sd, cot_sd


def visualize_results(results, save_dir='visualization', step=0):
    import os
    import wandb
    
    os.makedirs(save_dir, exist_ok=True)
    
    out_img_gen = results['out_img_gen']
    out_img_spe = results['out_img_spe']
    rec_lq = results['rec_lq']
    rec_hq = results['rec_hq']
    lq_img = results['lq_img']
    hq_img = results['hq_img']
    prompts_gen = results['prompts_gen']
    prompts_spe = results['prompts_spe']
    answer = results['answer']
    
    batch_size = out_img_gen.shape[0]
    
    # Prepare wandb logging data
    wandb_images = []
    wandb_table_data = []
    
    for i in range(batch_size):
        # Convert to images
        img_hq_gt = _tensor_to_uint8_hwc(hq_img[i])
        img_rec_hq = _tensor_to_uint8_hwc(rec_hq[i])
        img_lq_input = _tensor_to_uint8_hwc(lq_img[i])
        img_rec_lq = _tensor_to_uint8_hwc(rec_lq[i])
        img_output_gen = _tensor_to_uint8_hwc(out_img_gen[i])
        img_output_spe = _tensor_to_uint8_hwc(out_img_spe[i])
        
        # Save images locally
        prefix = f"step_{step}_sample_{i}"
        Image.fromarray(img_hq_gt).save(os.path.join(save_dir, f"{prefix}_hq_gt.png"))
        Image.fromarray(img_rec_hq).save(os.path.join(save_dir, f"{prefix}_rec_hq.png"))
        Image.fromarray(img_lq_input).save(os.path.join(save_dir, f"{prefix}_lq_input.png"))
        Image.fromarray(img_rec_lq).save(os.path.join(save_dir, f"{prefix}_rec_lq.png"))
        Image.fromarray(img_output_gen).save(os.path.join(save_dir, f"{prefix}_result_gen.png"))
        Image.fromarray(img_output_spe).save(os.path.join(save_dir, f"{prefix}_result_spe.png"))
        
        # Create visualization: [HQ GT] [LQ Input] [Out Gen] | [HQ GT] [LQ Input] [Out Spe]
        # Left side (gen prompt): [HQ GT] [LQ Input] [Out Gen]
        h, w = img_output_gen.shape[:2]
        left_side = np.zeros((h, w*3, 3), dtype=np.uint8)
        left_side[:, 0:w] = img_hq_gt
        left_side[:, w:2*w] = img_rec_hq
        left_side[:, 2*w:3*w] = img_lq_input
        
        # Right side (spe prompt): [HQ GT] [LQ Input] [Out Spe]
        right_side = np.zeros((h, w*3, 3), dtype=np.uint8)
        right_side[:, 0:w] = img_rec_lq
        right_side[:, w:2*w] = img_output_gen
        right_side[:, 2*w:3*w] = img_output_spe
        
        # Concatenate left and right sides horizontally
        comparison = np.concatenate([left_side, right_side], axis=1)
        
        # Log to wandb with image and text
        caption = (
            f"Sample {i} | "
            f"Gen: {prompts_gen[i][:80]}... | "
            f"Spe: {prompts_spe[i][:80]}... | "
            f"Answer: {answer[i][:80]}..."
        )
        wandb_images.append(
            wandb.Image(
                comparison,
                caption=caption
            )
        )
        
        # Prepare table data
        wandb_table_data.append([
            i,
            prompts_gen[i],
            prompts_spe[i],
            answer[i],
            wandb.Image(comparison)
        ])
    
    # Log images with error handling (non-blocking)
    try:
        wandb.log({"validation_images": wandb_images}, step=step)
    except Exception as e:
        print(f"Warning: Failed to log validation images to wandb: {e}")
        print("Note: Images are still saved locally. Training will continue.")
    
    # Log table with full text (with error handling - table logging can timeout)
    # Table logging is optional and can be skipped if network issues occur
    try:
        table = wandb.Table(
            columns=["Sample_ID", "Prompt_Gen", "Prompt_Spe", "Answer", "Comparison"],
            data=wandb_table_data
        )
        wandb.log({"validation_results": table}, step=step)
    except Exception as e:
        # Table logging often fails due to network timeouts, but this is non-critical
        print(f"Warning: Failed to log validation table to wandb (network timeout is common): {e}")
        print("Note: This is non-critical. Training will continue. Images are saved locally.")
    
    return


def save_checkpoint(model, optimizer, global_step, output_dir, checkpoints_total_limit=None, logger=None, lr_scheduler=None, lambda_param=None, lambda_optimizer=None):
    """
    Save model, optimizer, lr_scheduler, and lambda parameter checkpoints with automatic cleanup of old checkpoints.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        global_step: Current training step
        output_dir: Directory to save checkpoints
        checkpoints_total_limit: Maximum number of checkpoints to keep (None = keep all)
        logger: Logger for info messages
        lr_scheduler: Learning rate scheduler to save (optional)
        lambda_param: Lagrange multiplier parameter to save (optional)
        lambda_optimizer: Lambda optimizer to save (optional)
    """
    import shutil
    
    # Check and remove old checkpoints if limit is set
    if checkpoints_total_limit is not None:
        checkpoints = os.listdir(output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        
        if len(checkpoints) >= checkpoints_total_limit:
            num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            
            if logger:
                logger.info(
                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                )
                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
            
            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                shutil.rmtree(removing_checkpoint)
    
    # Save new checkpoint
    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(save_path, exist_ok=True)
    
    model_sd = _strip_ddp_prefix(model.state_dict())
    _, lora_sd, cot_sd = _split_state_dict_for_save(model_sd)
    torch.save(lora_sd, os.path.join(save_path, 'lora.pt'))
    torch.save(cot_sd, os.path.join(save_path, 'cot_adapter.pt'))
    torch.save(optimizer.state_dict(), os.path.join(save_path, 'optimizer.bin'))
    
    # Save lr_scheduler if provided
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(save_path, 'lr_scheduler.bin'))
    
    # Save lambda parameter if provided
    if lambda_param is not None:
        lambda_state_path = os.path.join(save_path, 'lambda_param.bin')
        if isinstance(lambda_param, torch.nn.ParameterDict):
            state_to_save = {key: value.data.cpu() for key, value in lambda_param.items()}
        elif isinstance(lambda_param, torch.nn.ParameterList):
            state_to_save = {str(idx): value.data.cpu() for idx, value in enumerate(lambda_param)}
        else:
            state_to_save = lambda_param.data.cpu()

        torch.save(state_to_save, lambda_state_path)
        if logger:
            if isinstance(state_to_save, dict):
                for key, tensor in state_to_save.items():
                    logger.info(f"Saved lambda parameter [{key}]: raw={tensor.item():.4f}")
            else:
                logger.info(f"Saved lambda parameter: raw={state_to_save.item():.4f}")
    
    # Save lambda optimizer if provided
    if lambda_optimizer is not None:
        torch.save(lambda_optimizer.state_dict(), os.path.join(save_path, 'lambda_optimizer.bin'))
        if logger:
            logger.info("Saved lambda optimizer state")
    
    if logger:
        logger.info(f"Saved state to {save_path}")
    else:
        print(f"Saved state to {save_path}")
    
    return save_path