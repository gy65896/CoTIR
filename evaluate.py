#!/usr/bin/env python3
"""Compute pyiqa metrics for method outputs against HQ (folder-level means)."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from cotir.util import patch_transformers_torch_load_safety_once

try:
    import icecream  # noqa: F401
except ImportError:
    icecream = None

try:
    from pyiqa import create_metric
except ImportError:
    create_metric = None


METRIC_SPECS: tuple[tuple[str, str, str], ...] = (
    ("clipiqa_plus", "clipiqa+", "nr"),
    ("qalign", "qalign", "nr"),
    ("liqe", "liqe", "nr"),
    ("maclip", "maclip", "nr"),
    ("clip_iqa", "clipiqa", "nr"),
    ("psnr", "psnr", "fr"),
    ("ssim", "ssim", "fr"),
    ("lpips", "lpips", "fr"),
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
SUMMARY_NAME = "metrics_summary.txt"


@dataclass(frozen=True)
class RunSpec:
    method: str
    image_dir: Path


class MetricEvaluator:
    def __init__(self, device: str) -> None:
        if create_metric is None:
            sys.exit("Please install pyiqa first: pip install pyiqa")
        if icecream is None:
            print("[warn] icecream is not installed; qalign may fail to load.", flush=True)
        self.device = device
        self._metrics: dict[str, torch.nn.Module] = {}

    def _get_metric(self, name: str) -> torch.nn.Module:
        if name not in self._metrics:
            self._metrics[name] = create_metric(name, device=self.device).eval()
        return self._metrics[name]

    def warmup(self) -> None:
        if any(py_name == "qalign" for _, py_name, _ in METRIC_SPECS):
            patch_transformers_torch_load_safety_once()
        for _, py_name, _ in METRIC_SPECS:
            self._get_metric(py_name)

    @staticmethod
    def _load_tensor(path: Path, device: str) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    @staticmethod
    def _to_float(value: torch.Tensor) -> float:
        return float(value.item() if value.numel() == 1 else value.mean().item())

    def compute(self, pred_path: Path, ref_path: Path) -> dict[str, float]:
        pred = self._load_tensor(pred_path, self.device)
        ref = self._load_tensor(ref_path, self.device)
        scores: dict[str, float] = {}
        with torch.no_grad():
            for key, py_name, kind in METRIC_SPECS:
                metric = self._get_metric(py_name)
                if kind == "fr":
                    value = self._to_float(metric(pred, ref))
                else:
                    value = self._to_float(metric(pred))
                if value != value:
                    raise ValueError(f"Metric {key} returned NaN for {pred_path.name}")
                scores[key] = value
        return scores

    def aggregate(self, pred_dir: Path, hq_dir: Path, method: str) -> dict[str, float]:
        pairs = match_image_pairs(pred_dir, hq_dir)
        if not pairs:
            raise RuntimeError(f"No matched image pairs found between {pred_dir} and {hq_dir}")

        sums = {key: 0.0 for key, _, _ in METRIC_SPECS}
        for pred_path, ref_path in tqdm(pairs, desc=method, ncols=100):
            scores = self.compute(pred_path, ref_path)
            for key, value in scores.items():
                sums[key] += value

        count = len(pairs)
        return {key: sums[key] / count for key in sums}


def list_images(folder: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.name] = path
    return images


def match_image_pairs(pred_dir: Path, hq_dir: Path) -> list[tuple[Path, Path]]:
    pred_images = list_images(pred_dir)
    hq_images = list_images(hq_dir)
    shared_names = sorted(set(pred_images) & set(hq_images))
    return [(pred_images[name], hq_images[name]) for name in shared_names]


def discover_methods(method_root: Path) -> list[RunSpec]:
    """
    Discover evaluation image folders.

    Supported layouts:
    1) `method_root` itself directly contains images (e.g. `.../precise/`).
    2) `method_root/<method>/` contains subfolders (e.g. `.../<method>/precise/`).
       In this case, we evaluate any immediate subfolder that contains images.
    """
    if list_images(method_root):
        return [RunSpec(method=method_root.name, image_dir=method_root)]

    runs: list[RunSpec] = []
    for method_dir in sorted(path for path in method_root.iterdir() if path.is_dir()):
        # Case 2.1: the child dir directly contains images
        if list_images(method_dir):
            runs.append(RunSpec(method=method_dir.name, image_dir=method_dir))
            continue

        # Case 2.2: images are inside one more level (e.g. precise/vague)
        for sub in sorted(path for path in method_dir.iterdir() if path.is_dir()):
            if list_images(sub):
                # Keep `method` as the upper-level directory name (e.g. CoTIR-FLUX.1-12B),
                # while the current run is evaluated on `sub/` (e.g. precise/vague).
                runs.append(RunSpec(method=method_dir.name, image_dir=sub))
    return runs


def format_metric_value(key: str, value: float) -> str:
    if key == "psnr":
        return f"{value:.2f}"
    return f"{value:.4f}"


def format_row(method: str, means: dict[str, float]) -> str:
    values = [method]
    values.extend(format_metric_value(key, means[key]) for key, _, _ in METRIC_SPECS)
    return " & ".join(values)


def write_summary(method: str, means: dict[str, float], out_txt: Path) -> None:
    header = "method & " + " & ".join(key for key, _, _ in METRIC_SPECS)
    content = header + "\n" + format_row(method, means) + "\n"
    out_txt.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate method outputs against HQ with pyiqa metrics.")
    parser.add_argument("--hq-dir", type=Path, required=True, help="Ground-truth HQ image folder")
    parser.add_argument("--method-root", type=Path, required=True, help="Folder containing result images")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    evaluator = MetricEvaluator(device=device)
    print(f"Device: {device}", flush=True)
    print("Warming up metrics...", flush=True)
    evaluator.warmup()

    method_runs = discover_methods(args.method_root)
    if not method_runs:
        sys.exit(f"No method folders with images found under {args.method_root}")

    for run in method_runs:
        means = evaluator.aggregate(run.image_dir, args.hq_dir, run.method)
        out_txt = run.image_dir / SUMMARY_NAME
        write_summary(run.method, means, out_txt)
        print(format_row(run.method, means), flush=True)
        print(f"Saved summary to {out_txt}", flush=True)


if __name__ == "__main__":
    main()
