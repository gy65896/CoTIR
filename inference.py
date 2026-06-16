#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import ImageFile

from cotir.model import load_model
from cotir.util import (
    load_image_as_tensor,
    load_infer_backends_by_model,
    prepare_infer_inputs_by_model,
    read_infer_prompt,
    resolve_inference_paths,
    sample_and_decode_by_model,
    tensor_to_pil_image,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-image CoTIR inference with split weights.")
    parser.add_argument("--config", type=str, default="configs/test_cotir-9b.yaml")
    parser.add_argument("--lq", type=str, required=True, help="LQ image path")
    parser.add_argument("--hq", type=str, default=None, help="Optional HQ image path, defaults to LQ")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt txt path or raw prompt text")
    parser.add_argument("--output_root", type=str, default="outputs")
    parser.add_argument("--name", type=str, default=None, help="Output image name, default from LQ filename")
    parser.add_argument("--max_long_edge", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None, help="Override config.training.num_steps")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible inference")
    return parser.parse_args()


def _resolve_prompt(prompt_arg: str, lq_path: Path, samples_dir: Path) -> str:
    if prompt_arg is None:
        return read_infer_prompt(None, lq_path, samples_dir)
    prompt_path = Path(prompt_arg).expanduser()
    if prompt_path.exists() and prompt_path.is_file():
        return read_infer_prompt(str(prompt_path.resolve()), lq_path, samples_dir)
    return prompt_arg


def _collect_pairs(lq_input: Path, hq_input: Path) -> List[Tuple[Path, Path]]:
    if lq_input.is_dir() and hq_input.is_dir():
        pairs: List[Tuple[Path, Path]] = []
        lq_files = sorted(
            [p for p in lq_input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES],
            key=lambda p: p.name,
        )
        for lq_file in lq_files:
            hq_file = hq_input / lq_file.name
            if hq_file.exists() and hq_file.is_file():
                pairs.append((lq_file, hq_file))
            else:
                print(f"skip (hq not found): {hq_file}")
        return pairs

    return [(lq_input, hq_input)]


def _resize_roundtrip_inputs(hq: torch.Tensor, lq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], tuple[int, int]]:
    # Keep original size for final restore, but run inference at a safe multiple-of-16 size.
    orig_hw = (int(lq.shape[-2]), int(lq.shape[-1]))
    work_h = max(16, (orig_hw[0] // 16) * 16)
    work_w = max(16, (orig_hw[1] // 16) * 16)
    work_hw = (work_h, work_w)
    if work_hw != orig_hw:
        hq = F.interpolate(hq, size=work_hw, mode="bicubic", align_corners=False)
        lq = F.interpolate(lq, size=work_hw, mode="bicubic", align_corners=False)
    return hq, lq, orig_hw, work_hw


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    cfg = OmegaConf.load(Path(args.config).resolve())
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "none": torch.float32,
    }
    weight_dtype = dtype_map.get(str(cfg.get("mixed_precision", "bf16")).lower(), torch.float32)

    base_ckpt, cot_ckpt, lora_ckpt = resolve_inference_paths(cfg)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True

    lq_path = Path(args.lq).expanduser().resolve()
    hq_path = Path(args.hq).expanduser().resolve() if args.hq else lq_path
    samples_dir = Path(__file__).resolve().parent / "samples"

    t5, clip, ae = load_infer_backends_by_model(cfg=cfg, device=device, base_ckpt=base_ckpt)

    model = load_model(
        cfg.model,
        checkpoint_path=None,
        cot_adapter_checkpoint_path=str(cot_ckpt),
        lora_checkpoint_path=str(lora_ckpt),
        base_checkpoint_path=str(base_ckpt),
        device="cpu",
        verbose=False,
        load_weights=True,
        for_inference=True,
        merge_lora_inference=bool(getattr(getattr(cfg.model, "lora", None), "merge_for_inference", False)),
    )
    model.to(device, dtype=weight_dtype)
    model.eval()

    pairs = _collect_pairs(lq_path, hq_path)
    if not pairs:
        raise FileNotFoundError(f"No valid image pairs found from lq={lq_path} and hq={hq_path}")

    if not hasattr(cfg, "training"):
        cfg.training = OmegaConf.create({})
    if args.num_steps is not None:
        cfg.training.num_steps = int(args.num_steps)
    elif not hasattr(cfg.training, "num_steps"):
        cfg.training.num_steps = 20

    out_root = Path(args.output_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for lq_file, hq_file in pairs:
        prompt = _resolve_prompt(args.prompt, lq_file, samples_dir)
        hq = load_image_as_tensor(hq_file, args.max_long_edge).unsqueeze(0)
        lq = load_image_as_tensor(lq_file, args.max_long_edge).unsqueeze(0)
        hq, lq, orig_hw, _ = _resize_roundtrip_inputs(hq, lq)
        data = prepare_infer_inputs_by_model(
            base_model_name=cfg.model.base_model_name,
            hq_img=hq,
            lq_img=lq,
            prompt=prompt,
            ae=ae,
            t5=t5,
            clip=clip,
            device=device,
        )
        out_img = sample_and_decode_by_model(
            model=model,
            ae=ae,
            infer_data=data,
            hq_img=hq,
            base_model_name=cfg.model.base_model_name,
            num_steps=cfg.training.num_steps,
            device=device,
            weight_dtype=weight_dtype,
        )
        if tuple(out_img.shape[-2:]) != orig_hw:
            out_img = F.interpolate(out_img, size=orig_hw, mode="bicubic", align_corners=False)

        if args.name and len(pairs) == 1:
            name = args.name
        else:
            name = lq_file.name
        if not name.lower().endswith(".png"):
            name = f"{Path(name).stem}.png"
        out_path = out_root / name
        tensor_to_pil_image(out_img[0]).save(out_path)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
