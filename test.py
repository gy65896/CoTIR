import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageFile
from tqdm import tqdm

from cotir.model import load_model
from cotir.util import (
    load_image_as_tensor,
    load_infer_backends_by_model,
    resolve_inference_paths,
    tensor_to_pil_image,
    val_one_step,
)


Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_JSON = "./test/test.jsonl"
DEFAULT_LQ_DIR = "./test/lq"
DEFAULT_RESULTS = "./results"
MODEL_CONFIG_CHOICES = {
    "CoTIR-4B": "configs/test_cotir-4b.yaml",
    "CoTIR-9B": "configs/test_cotir-9b.yaml",
    "CoTIR-12B": "configs/test_cotir-12b.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CoTIR inference (gen vs special prompts).")
    parser.add_argument("--model", type=str, default=None, choices=list(MODEL_CONFIG_CHOICES.keys()), help="Preset model config")
    parser.add_argument("--config", type=str, default="configs/test_cotir-9b.yaml", help="Config file path")
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON, help="JSONL path with deg_type metadata")
    parser.add_argument("--lq_dir", type=str, default=DEFAULT_LQ_DIR, help="Directory containing LQ images")
    parser.add_argument("--output_root", type=str, default=DEFAULT_RESULTS, help="Output root directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    parser.add_argument("--max_long_edge", type=int, default=None, help="Resize image long edge to this value")
    parser.add_argument("--max_samples", type=int, default=None, help="Only process first N samples")
    parser.add_argument("--device", type=str, default=None, help="Runtime device, default auto cuda->cpu")
    parser.add_argument("--skip_existing", action="store_true", help="Skip if both gen/special output files already exist")
    return parser.parse_args()


def load_image(path: Path, max_long_edge: int | None = None) -> torch.Tensor:
    return load_image_as_tensor(path, max_long_edge)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return tensor_to_pil_image(tensor)


@dataclass
class SampleBatch:
    entries: List[dict]
    hq: torch.Tensor
    lq: torch.Tensor
    prompts_gen: List[str]
    prompts_spe: List[str]
    answers_s: List[str]
    answers_d: List[str]
    answers_p: List[str]


class InferenceRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        config_path = MODEL_CONFIG_CHOICES[args.model] if args.model else args.config
        self.config_path = Path(config_path).resolve()
        self.config = OmegaConf.load(self.config_path)

        self.device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "none": torch.float32,
        }
        mixed_precision = str(self.config.get("mixed_precision", "bf16")).lower()
        self.weight_dtype = dtype_map.get(mixed_precision, torch.float32)

        self.json_path = Path(args.json_path).resolve()
        self.lq_dir = Path(args.lq_dir).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.gen_dir = self.output_root / "vague"
        self.spe_dir = self.output_root / "precise"
        self.gen_dir.mkdir(parents=True, exist_ok=True)
        self.spe_dir.mkdir(parents=True, exist_ok=True)

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        base_ckpt, cot_ckpt, lora_ckpt = resolve_inference_paths(self.config)
        self.t5, self.clip, self.ae = load_infer_backends_by_model(
            cfg=self.config,
            device=self.device,
            base_ckpt=base_ckpt,
        )

        self.model = load_model(
            self.config.model,
            checkpoint_path=None,
            base_checkpoint_path=str(base_ckpt),
            cot_adapter_checkpoint_path=str(cot_ckpt),
            lora_checkpoint_path=str(lora_ckpt),
            device="cpu",
            verbose=False,
            for_inference=True,
            merge_lora_inference=bool(getattr(getattr(self.config.model, "lora", None), "merge_for_inference", False)),
        )
        self.model.to(self.device, dtype=self.weight_dtype)
        self.model.eval()

    def _load_manifest(self) -> List[dict]:
        records: List[dict] = []
        with self.json_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                records.append(json.loads(line))
                if self.args.max_samples and len(records) >= self.args.max_samples:
                    break
        return records

    def _resolve_image_path(self, entry: dict) -> Path:
        path_str = entry.get("lq_path", "")
        if path_str:
            candidate = Path(path_str)
            if candidate.is_absolute() and candidate.exists():
                return candidate
            rel_to_json = (self.json_path.parent / candidate).resolve()
            if rel_to_json.exists():
                return rel_to_json
            fname = candidate.name
        else:
            fname = f"{entry.get('id', 'sample')}.png"
        direct = self.lq_dir / fname
        if direct.exists():
            return direct
        raise FileNotFoundError(f"Image not found at either path: {path_str} or {direct}")

    def _prepare_batch(self, entries: List[dict]) -> SampleBatch:
        hq_tensors = []
        lq_tensors = []
        prompts_gen = []
        prompts_spe = []
        answers_s = []
        answers_d = []
        answers_p = []

        for entry in entries:
            img_path = self._resolve_image_path(entry)
            lq_tensor = load_image(img_path, self.args.max_long_edge)
            # Use LQ as placeholder when HQ is unavailable.
            hq_tensor = lq_tensor.clone()

            deg_type = entry.get("deg_type") or ""
            prompt_gen = entry.get("gen_prompt")
            prompt_spe = entry.get("spe_prompt")

            if not isinstance(prompt_gen, str) or not prompt_gen.strip():
                raise ValueError(f"Sample {entry.get('id', 'unknown')} is missing `gen_prompt`.")
            if not isinstance(prompt_spe, str) or not prompt_spe.strip():
                raise ValueError(f"Sample {entry.get('id', 'unknown')} is missing `spe_prompt`.")

            hq_tensors.append(hq_tensor)
            lq_tensors.append(lq_tensor)
            prompts_gen.append(prompt_gen)
            prompts_spe.append(prompt_spe)
            answers_s.append(entry.get("subset", ""))
            answers_d.append(deg_type)
            answers_p.append(prompt_spe)

        hq_batch = torch.stack(hq_tensors, dim=0)
        lq_batch = torch.stack(lq_tensors, dim=0)

        return SampleBatch(
            entries=entries,
            hq=hq_batch,
            lq=lq_batch,
            prompts_gen=prompts_gen,
            prompts_spe=prompts_spe,
            answers_s=answers_s,
            answers_d=answers_d,
            answers_p=answers_p,
        )

    def _save_outputs(self, batch: SampleBatch, results: dict) -> None:
        gen_imgs = results["out_img_gen"]
        spe_imgs = results["out_img_spe"]

        for entry, gen_tensor, spe_tensor in zip(batch.entries, gen_imgs, spe_imgs):
            base_name = Path(entry.get("lq_path", f"{entry.get('id', 'sample')}.png")).name
            gen_path = self.gen_dir / base_name
            spe_path = self.spe_dir / base_name

            if self.args.skip_existing and gen_path.exists() and spe_path.exists():
                continue

            tensor_to_pil(gen_tensor).save(gen_path)
            tensor_to_pil(spe_tensor).save(spe_path)

    def run(self) -> None:
        records = self._load_manifest()
        if not records:
            print(f"No samples were loaded from {self.json_path}.")
            return

        total = len(records)
        progress = tqdm(total=total, desc="Inference", unit="img")
        pending: List[dict] = []

        for entry in records:
            base_name = Path(entry.get("lq_path", f"{entry.get('id', 'sample')}.png")).name
            gen_exists = (self.gen_dir / base_name).exists()
            spe_exists = (self.spe_dir / base_name).exists()
            if self.args.skip_existing and gen_exists and spe_exists:
                progress.update(1)
                continue

            pending.append(entry)
            if len(pending) == self.args.batch_size:
                self._process_pending(pending)
                progress.update(len(pending))
                pending = []

        if pending:
            self._process_pending(pending)
            progress.update(len(pending))

        progress.close()
        print(f"Inference completed. Results saved to {self.gen_dir} / {self.spe_dir}.")

    def _process_pending(self, entries: Sequence[dict]) -> None:
        batch = self._prepare_batch(list(entries))
        data_batch = (
            batch.hq,
            batch.lq,
            batch.prompts_gen,
            batch.prompts_spe,
            batch.answers_s,
            batch.answers_d,
            batch.answers_p,
        )
        results = val_one_step(
            model=self.model,
            batch=data_batch,
            args=self.config,
            t5=self.t5,
            clip=self.clip,
            ae=self.ae,
            device=self.device,
            weight_dtype=self.weight_dtype,
            use_gen_prob=1.0,
        )
        self._save_outputs(batch, results)


def main():
    args = parse_args()
    runner = InferenceRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

