import os
from pathlib import Path
from torch import Tensor, nn
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class HFEmbedder(nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int,
        is_clip: bool | None = None,
        local_files_only: bool = False,
        tokenizer_source: str | None = None,
        **hf_kwargs,
    ):
        super().__init__()
        self.is_clip = version.startswith("openai") if is_clip is None else bool(is_clip)
        self.max_length = max_length
        self.output_key = "pooler_output" if self.is_clip else "last_hidden_state"
        tokenizer_source = tokenizer_source or version

        # Get cache directory
        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or os.path.expanduser("~/.cache/huggingface")
        model_cache_dir = Path(cache_dir) / "hub" / f"models--{version.replace('/', '--')}"
        
        if self.is_clip:
            print(f"Loading CLIP checkpoint: {model_cache_dir}")
            self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
                tokenizer_source, max_length=max_length, local_files_only=local_files_only
            )
            self.hf_module: CLIPTextModel = CLIPTextModel.from_pretrained(
                version, local_files_only=local_files_only, **hf_kwargs
            )
        else:
            print(f"Loading T5 checkpoint: {model_cache_dir}")
            self.tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(
                tokenizer_source, max_length=max_length, local_files_only=local_files_only, legacy=False
            )
            self.hf_module: T5EncoderModel = T5EncoderModel.from_pretrained(
                version, local_files_only=local_files_only, **hf_kwargs
            )

        self.hf_module = self.hf_module.eval().requires_grad_(False)

    def forward(self, text: list[str]) -> Tensor:
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        outputs = self.hf_module(
            input_ids=batch_encoding["input_ids"].to(self.hf_module.device),
            attention_mask=None,
            output_hidden_states=False,
        )
        return outputs[self.output_key].bfloat16()
