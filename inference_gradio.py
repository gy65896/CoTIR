#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio single-image inference UI.
"""

import os
import random
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageFile
from omegaconf import OmegaConf

from cotir.flux.util import configs as FLUX_MODEL_CONFIGS, load_ae as load_ae_flux, load_clip, load_t5
from cotir.flux2.util import FLUX2_MODEL_INFO, load_ae as load_ae_flux2, load_text_encoder
from cotir.model import load_model
from cotir.util import (
    load_pil_image_as_tensor,
    prepare_infer_inputs_by_model,
    resolve_inference_paths,
    sample_and_decode_by_model,
    tensor_to_pil_image,
)

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

MODEL_CONFIG_CHOICES = {
    "CoTIR-4B": "configs/test_cotir-4b.yaml",
    "CoTIR-9B": "configs/test_cotir-9b.yaml",
    "CoTIR-12B": "configs/test_cotir-12b.yaml",
}

def find_available_port(start_port: int = 7860, max_tries: int = 20) -> int:
    """
    Find an available TCP port starting from start_port.
    """
    import socket

    port = start_port
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                port += 1
                continue
    raise RuntimeError(f"No free port found from {start_port} within {max_tries} tries.")

def gamma_correct_illumination(img, variance, gamma=2.2, eps=1e-6):
    """
    Gamma correct the illumination component then multiply back.
    """
    img_float = np.float64(img)
    illumination = cv2.GaussianBlur(img_float, (0, 0), variance)
    illumination_normalized = illumination / 255.0
    illumination_corrected = np.power(illumination_normalized, 1.0 / gamma) * 255.0
    ratio = illumination_corrected / (illumination + eps)
    result = img_float * ratio
    result = np.clip(result, 0, 255)
    result = np.uint8(result)
    return result


def SSR_gamma(img, variance, gamma=2.2):
    """
    Single-scale Retinex with gamma correction on illumination.
    """
    result = gamma_correct_illumination(img, variance, gamma)
    return np.uint8(result)


def white_balance_gray_world(img):
    """
    Gray-world white balance.
    """
    img_float = img.astype(np.float64)
    avg_b = np.mean(img_float[:, :, 0])
    avg_g = np.mean(img_float[:, :, 1])
    avg_r = np.mean(img_float[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    
    gain_b = avg_gray / (avg_b + 1e-6)
    gain_g = avg_gray / (avg_g + 1e-6)
    gain_r = avg_gray / (avg_r + 1e-6)
    
    result = img_float.copy()
    result[:, :, 0] = np.clip(img_float[:, :, 0] * gain_b, 0, 255)
    result[:, :, 1] = np.clip(img_float[:, :, 1] * gain_g, 0, 255)
    result[:, :, 2] = np.clip(img_float[:, :, 2] * gain_r, 0, 255)
    
    return result.astype(np.uint8)


def apply_postprocess(pil_img: Image.Image, enable_enhancement: bool, enable_white_balance: bool,
                      gamma: float = 1.5, variance: float = 100) -> Image.Image:
    """
    Optional gamma enhancement and white balance on a PIL image.
    """
    if not enable_enhancement and not enable_white_balance:
        return pil_img
    
    # PIL RGB -> OpenCV BGR
    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    
    if enable_enhancement:
        img_bgr = SSR_gamma(img_bgr, variance, gamma)
    
    if enable_white_balance:
        img_bgr = white_balance_gray_world(img_bgr)
    
    # OpenCV BGR -> PIL RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_image_from_path(image_path: str) -> Image.Image:
    """Load image from path."""
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    img = Image.open(path)
    return img


def _load_prompt_for_image(image_path: str, fallback_prompt: str = "") -> str:
    p = Path(image_path).expanduser().resolve()
    prompt_path = p.parent.parent / "prompts" / f"{p.stem}.txt"
    if prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return fallback_prompt


def pad_to_multiple_of_16(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Resize image so both sides are divisible by 16.
    """
    original_size = img.size  # (width, height)
    width, height = original_size
    
    if width % 16 == 0 and height % 16 == 0:
        return img, original_size
    
    new_width = (width // 16) * 16
    new_height = (height // 16) * 16
    
    if new_width == 0:
        new_width = 16
    if new_height == 0:
        new_height = 16
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    return resized_img, original_size


def restore_original_size(img: Image.Image, original_size: Tuple[int, int]) -> Image.Image:
    """
    Resize back to original size if it changed.
    """
    if img.size == original_size:
        return img
    return img.resize(original_size, Image.LANCZOS)


def resolve_checkpoint(cfg, explicit_ckpt: str | None, cfg_path: Path) -> str | None:
    if explicit_ckpt:
        p = Path(explicit_ckpt).expanduser().resolve()
        if p.is_dir():
            p = p / "model.bin"
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return str(p)

    output_dir_cfg = Path(cfg.data.output_dir)
    candidates_dir = []
    if not output_dir_cfg.is_absolute():
        candidates_dir.append((cfg_path.parent / output_dir_cfg).resolve())
        candidates_dir.append((cfg_path.parent.parent / output_dir_cfg).resolve())
    else:
        candidates_dir.append(output_dir_cfg)

    output_dir = None
    tried = []
    for cand in candidates_dir:
        tried.append(str(cand))
        if cand.exists():
            output_dir = cand
            break
    if output_dir is None:
        print(f"[info] checkpoint directory not found, tried: {tried}, use pretrained.")
        return None

    ckpt_folders = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not ckpt_folders:
        print(f"[info] no checkpoint in {output_dir}, use pretrained.")
        return None

    latest = sorted(ckpt_folders, key=lambda x: int(x.name.split("-")[1]))[-1]
    model_path = latest / "model.bin"
    if not model_path.exists():
        print(f"[warn] {latest} missing model.bin, fallback to pretrained.")
        return None
    print(f"Use checkpoint: {model_path}")
    return str(model_path)


@dataclass
class SampleBatch:
    entries: List[dict]
    hq: torch.Tensor
    lq: torch.Tensor
    prompts: List[str]


class GlobalState:
    def __init__(self):
        self.model = None
        self.t5 = None
        self.clip = None
        self.ae = None
        self.device = torch.device("cpu")
        self.weight_dtype = torch.float32
        self.config = None
        self.output_root = Path("./outputs").resolve()


STATE = GlobalState()


# ========= Core helpers =========
def _unload_models():
    STATE.model = None
    STATE.t5 = None
    STATE.clip = None
    STATE.ae = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def prepare_single_sample(pil_img_lq: Image.Image, pil_img_hq: Image.Image | None, prompt: str, max_long_edge=None) -> SampleBatch:
    lq_tensor = load_pil_image_as_tensor(pil_img_lq, max_long_edge, keep_multiple_of_16=True)
    if pil_img_hq is not None:
        hq_tensor = load_pil_image_as_tensor(pil_img_hq, max_long_edge, keep_multiple_of_16=True)
    else:
        hq_tensor = lq_tensor.clone()
    return SampleBatch(
        entries=[{"lq_path": "uploaded_image"}],
        hq=torch.stack([hq_tensor]),
        lq=torch.stack([lq_tensor]),
        prompts=[prompt],
    )


def init_models(cuda_devices: str, config_path: str):
    # Reload safety: unload previous model stack first.
    _unload_models()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg_path = Path(config_path).expanduser().resolve()
    config = OmegaConf.load(cfg_path)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": torch.float32,
    }
    mixed_precision = str(config.get("mixed_precision", "bf16")).lower()
    weight_dtype = dtype_map.get(mixed_precision, torch.float32)

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    base_model_name = str(config.model.base_model_name).lower()
    is_flux2 = base_model_name.startswith("flux.2")
    base_ckpt, cot_ckpt, lora_ckpt = resolve_inference_paths(config)

    if is_flux2:
        flux2_root = Path(str(config.inference.base_model_path)).expanduser().resolve()
        os.environ["FLUX2_MODEL_DIR"] = str(flux2_root)
        t5 = load_text_encoder(config.model.base_model_name, device=device).requires_grad_(False).eval()
        clip = None
        ae = load_ae_flux2(config.model.base_model_name, device=device).requires_grad_(False).eval()
    else:
        kontext_root = Path(str(config.inference.base_model_path)).expanduser().resolve()
        os.environ["FLUX_T5_PATH"] = str(kontext_root / "text_encoder_2")
        os.environ["FLUX_T5_TOKENIZER_PATH"] = str(kontext_root / "tokenizer_2")
        os.environ["FLUX_CLIP_PATH"] = str(kontext_root / "text_encoder")
        os.environ["FLUX_CLIP_TOKENIZER_PATH"] = str(kontext_root / "tokenizer")
        os.environ["FLUX_AE"] = str(kontext_root / "ae.safetensors")
        t5 = load_t5(device, max_length=512).requires_grad_(False).eval()
        clip = load_clip(device).requires_grad_(False).eval()
        ae = load_ae_flux(config.model.base_model_name, device=device).requires_grad_(False).eval()

    model = load_model(
        config.model,
        checkpoint_path=None,
        base_checkpoint_path=str(base_ckpt),
        cot_adapter_checkpoint_path=str(cot_ckpt),
        lora_checkpoint_path=str(lora_ckpt),
        device="cpu",
        verbose=False,
        for_inference=True,
        merge_lora_inference=bool(getattr(getattr(config.model, "lora", None), "merge_for_inference", False)),
    ).requires_grad_(False)
    model.to(device, dtype=weight_dtype)
    model.eval()

    STATE.model = model
    STATE.t5 = t5
    STATE.clip = clip
    STATE.ae = ae
    STATE.device = device
    STATE.weight_dtype = weight_dtype
    STATE.config = config

    return (
        f"Model loaded. device: {device}, precision: {mixed_precision}\n"
        f"base={base_ckpt}\ncot={cot_ckpt}\nlora={lora_ckpt}"
    )


def run_inference(image_path: str, hq_path: str, prompt: str,
                  output_root: str, max_edge: int | None,
                  enable_enhancement: bool, enable_white_balance: bool,
                  gamma: float, variance: float,
                  num_steps: int, seed: int):
    t0 = time.perf_counter()
    batch = infer_data = x_0 = img = guidance_vec = None
    img_ids = img_cond_ids = txt = txt_ids = vec = x_c = out_img = None
    if STATE.model is None:
        raise RuntimeError("Please initialize/reload model first.")
    if not image_path or not image_path.strip():
        raise RuntimeError("Please provide image path.")
    prompt = _load_prompt_for_image(image_path, fallback_prompt=str(prompt or "").strip())
    if not prompt:
        raise RuntimeError("Prompt is empty and no matched prompt txt file found.")

    try:
        pil_img_lq_original = load_image_from_path(image_path.strip())
    except Exception as e:
        raise RuntimeError(f"Load LQ image failed: {e}")

    pil_img_lq, original_size = pad_to_multiple_of_16(pil_img_lq_original)
    size_changed = (pil_img_lq.size != original_size)

    pil_img_hq = None
    pil_img_hq_original = None
    if hq_path and hq_path.strip():
        try:
            pil_img_hq_original = load_image_from_path(hq_path.strip())
            pil_img_hq = pil_img_hq_original
            if size_changed:
                pil_img_hq = pil_img_hq.resize(pil_img_lq.size, Image.LANCZOS)
        except Exception as e:
            print(f"[warn] Load HQ image failed: {e}, fallback to LQ only.")

    out_root_path = Path(str(output_root)).expanduser().resolve()
    max_edge_int = None
    try:
        if max_edge not in (None, ""):
            max_edge_int = int(max_edge)
    except Exception:
        max_edge_int = None

    try:
        batch = prepare_single_sample(pil_img_lq, pil_img_hq, prompt, max_edge_int)

        infer_data = prepare_infer_inputs_by_model(
            base_model_name=STATE.config.model.base_model_name,
            hq_img=batch.hq.to(STATE.device),
            lq_img=batch.lq.to(STATE.device),
            prompt=prompt,
            ae=STATE.ae,
            t5=STATE.t5,
            clip=STATE.clip,
            device=STATE.device,
        )

        out_img = sample_and_decode_by_model(
            model=STATE.model,
            ae=STATE.ae,
            infer_data=infer_data,
            hq_img=batch.hq.to(STATE.device),
            base_model_name=STATE.config.model.base_model_name,
            num_steps=int(num_steps),
            device=STATE.device,
            weight_dtype=STATE.weight_dtype,
            seed=int(seed),
        )
        out_pil = tensor_to_pil_image(out_img[0])

        if size_changed:
            out_pil = restore_original_size(out_pil, original_size)

        if enable_enhancement or enable_white_balance:
            out_pil = apply_postprocess(out_pil, enable_enhancement, enable_white_balance, gamma, variance)
            postprocess_info = f"\npostprocess: enhance={enable_enhancement}, white_balance={enable_white_balance}, gamma={gamma}, variance={variance}"
        else:
            postprocess_info = ""

        out_root_path.mkdir(parents=True, exist_ok=True)
        out_dir = out_root_path
        out_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(image_path.strip())
        base_name = input_path.stem + ".png"
        out_path = out_dir / base_name
        out_pil.save(out_path)

        size_info = ""
        if size_changed:
            size_info = f"\nsize: {original_size} -> {pil_img_lq.size} -> {original_size}"

        elapsed_s = time.perf_counter() - t0
        info = f"saved:\nOUT -> {out_path}\nsteps={num_steps}, seed={seed}, max_edge={max_edge_int}, time={elapsed_s:.3f}s{postprocess_info}{size_info}"
        return pil_img_lq_original, pil_img_hq_original, out_pil, info
    finally:
        batch = infer_data = None
        out_img = None
        if torch.cuda.is_available() and STATE.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# ========= Gradio UI =========
def build_app():
    with gr.Blocks(title="CoTIR Single-Image Inference") as demo:
        gr.Markdown("## CoTIR Single-Image Inference (Gradio)")

        with gr.Accordion("Init / Reload Model", open=True):
            gpu = gr.Textbox(label="CUDA_VISIBLE_DEVICES", value="0")
            model_choice = gr.Dropdown(
                label="Model",
                choices=list(MODEL_CONFIG_CHOICES.keys()),
                value="CoTIR-12B",
            )
            init_btn = gr.Button("Init / Reload")
            init_log = gr.Textbox(label="Init Log", lines=4)

        with gr.Row():
            with gr.Column():
                img_input = gr.Textbox(label="Input image path (LQ)", placeholder="/path/to/lq_image.png", value="./samples/images/000018.png")
                hq_input = gr.Textbox(label="HQ image path (optional)", placeholder="/path/to/hq_image.png", value="")
                prompt_box = gr.Textbox(label="Prompt (optional, auto from matched txt if empty)", value="")
                
                with gr.Accordion("Inference & Postprocess", open=True):
                    with gr.Row():
                        output_root = gr.Textbox(label="Output Root", value="./outputs")
                        max_edge = gr.Number(label="Max long edge (optional)", value=2560, precision=0)
                    with gr.Row():
                        num_steps = gr.Slider(label="Sampling steps (num_steps)", minimum=1, maximum=50, value=5, step=1)
                        infer_seed = gr.Number(label="Inference seed", value=42, precision=0)
                    with gr.Row():
                        enable_enhancement = gr.Checkbox(label="Enable gamma enhancement", value=False)
                        enable_white_balance = gr.Checkbox(label="Enable white balance", value=False)
                    with gr.Accordion("Gamma options", open=False):
                        with gr.Row():
                            gamma_value = gr.Slider(label="Gamma (>1 brighten)", minimum=0.5, maximum=3.0, value=1.5, step=0.1, visible=False)
                            variance_value = gr.Slider(label="Gaussian blur sigma", minimum=10, maximum=200, value=100, step=10, visible=False)
                
                run_btn = gr.Button("Run Inference")
                save_log = gr.Textbox(label="Info", lines=4)

        with gr.Row():
            lq_display = gr.Image(label="Input (LQ)", type="pil")
            hq_display = gr.Image(label="Reference (HQ)", type="pil")
        
        with gr.Row():
            out_img = gr.Image(label="Output", type="pil")

        def _init_wrapper(gpu, model_choice):
            config_path = MODEL_CONFIG_CHOICES[model_choice]
            msg = init_models(
                cuda_devices=gpu,
                config_path=config_path,
            )
            return msg

        init_btn.click(
            _init_wrapper,
            inputs=[gpu, model_choice],
            outputs=init_log,
        )

        def _toggle_gamma_controls(enable):
            return [
                gr.update(visible=enable),
                gr.update(visible=enable),
            ]

        enable_enhancement.change(
            _toggle_gamma_controls,
            inputs=enable_enhancement,
            outputs=[gamma_value, variance_value],
        )

        run_btn.click(
            run_inference,
            inputs=[
                img_input,
                hq_input,
                prompt_box,
                output_root,
                max_edge,
                enable_enhancement,
                enable_white_balance,
                gamma_value,
                variance_value,
                num_steps,
                infer_seed,
            ],
            outputs=[lq_display, hq_display, out_img, save_log],
        )

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862, help="server port")
    parser.add_argument("--share", action="store_true", help="whether to create public link")
    args = parser.parse_args()
    
    try:
        port = find_available_port(args.port)
        if port != args.port:
            print(f"[info] Port {args.port} busy, fallback to {port}")
    except Exception as e:
        raise RuntimeError(f"Failed to find free port near {args.port}: {e}")

    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=port, share=args.share)

