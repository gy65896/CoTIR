import base64
import io
import os
import sys

import torch
from PIL import Image
from safetensors.torch import load_file as load_sft

from .autoencoder import AutoEncoder, AutoEncoderParams
from .model import Flux2, Flux2Params, Klein4BParams, Klein9BParams
from .text_encoder import Flux2Qwen3SubfolderEmbedder, load_mistral_small_embedder, load_qwen3_embedder

FLUX2_MODEL_INFO = {
    "flux.2-klein-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        # Prefer the bundled text encoder under the FLUX.2 repo/model-dir (bf16), like debug.py.
        "text_encoder_load_fn": lambda device="cuda": Flux2Qwen3SubfolderEmbedder(
            flux2_model_dir_or_repo="black-forest-labs/FLUX.2-klein-4B",
            device=device,
            torch_dtype=torch.bfloat16,
        ),
        "model_path": "KLEIN_4B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
    },
    "flux.2-klein-9b-kv": {
        "repo_id": "black-forest-labs/FLUX.2-klein-9B-kv",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-9b-kv.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_KV_MODEL_PATH",
        "defaults": {"guidance": 1.0, "num_steps": 4},
        "fixed_params": {"guidance", "num_steps"},  # guidance and timestep distilled
        "guidance_distilled": True,
        "use_kv_cache": True,
    },
    "flux.2-klein-base-4b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-4B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-4b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein4BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="4B", device=device),
        "model_path": "KLEIN_4B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-klein-base-9b": {
        "repo_id": "black-forest-labs/FLUX.2-klein-base-9B",
        "ae_repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux-2-klein-base-9b.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Klein9BParams(),
        "text_encoder_load_fn": lambda device="cuda": load_qwen3_embedder(variant="8B", device=device),
        "model_path": "KLEIN_9B_BASE_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": False,
    },
    "flux.2-dev": {
        "repo_id": "black-forest-labs/FLUX.2-dev",
        "filename": "flux2-dev.safetensors",
        "filename_ae": "ae.safetensors",
        "params": Flux2Params(),
        "text_encoder_load_fn": load_mistral_small_embedder,
        "model_path": "FLUX2_MODEL_PATH",
        "defaults": {"guidance": 4.0, "num_steps": 50},
        "fixed_params": {},
        "guidance_distilled": True,
    },
}

def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(unexpected))


class _DiffusersFlux2AEAdapter(torch.nn.Module):
    """Wrap diffusers AutoencoderKLFlux2 to match CoTIR `encode` / `decode` latent layout."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(b, c * 4, h // 2, w // 2)
        return latents

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(b, c // 4, h * 2, w * 2)
        return latents

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = self.vae.encode(x).latent_dist.mode()
        latents = self._patchify_latents(latents)

        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = (latents - bn_mean) / bn_std
        return latents

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.to(device=self.vae.device, dtype=self.vae.dtype)
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(z.device, z.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(
            z.device, z.dtype
        )
        z = z * bn_std + bn_mean
        z = self._unpatchify_latents(z)
        img = self.vae.decode(z, return_dict=False)[0]
        return img


def _require_local_dir(model_name: str, *, check_flow_weights: bool = True) -> str:
    """
    Force local-only loading (no downloads).
    Set env `FLUX2_MODEL_DIR` or default to the workspace path used in debug.py.
    """
    local_dir = os.environ.get("FLUX2_MODEL_DIR", "/forest/timmy/rebuttal/FLUX.2-klein-4B")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"Local FLUX.2 model dir not found: {local_dir}. "
            f"Please set env FLUX2_MODEL_DIR to your local model folder."
        )
    # Minimal sanity: ensure the expected weight file exists for klein-4b (when loading flow weights).
    if check_flow_weights and model_name.lower() == "flux.2-klein-4b":
        expected = os.path.join(local_dir, "flux-2-klein-4b.safetensors")
        if not os.path.exists(expected):
            raise FileNotFoundError(f"Missing local weight file: {expected}")
    return local_dir


def load_flow_model(model_name: str, debug_mode: bool = False, device: str | torch.device = "cuda") -> Flux2:
    config = FLUX2_MODEL_INFO[model_name.lower()]

    if debug_mode:
        config["params"].depth = 1
        config["params"].depth_single_blocks = 1
    else:
        local_dir = _require_local_dir(model_name)
        weight_path = os.path.join(local_dir, config["filename"])

    if not debug_mode:
        with torch.device("meta"):
            model = Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)
        print(f"Loading {weight_path} for the FLUX.2 weights")
        sd = load_sft(weight_path, device=str(device))
        model.load_state_dict(sd, strict=True, assign=True)
        return model.to(device)
    else:
        with torch.device(device):
            return Flux2(FLUX2_MODEL_INFO[model_name.lower()]["params"]).to(torch.bfloat16)


def load_text_encoder(
    model_name: str, device: str | torch.device = "cuda", load_weights: bool = True
):
    config = FLUX2_MODEL_INFO[model_name.lower()]
    local_dir = _require_local_dir(model_name, check_flow_weights=load_weights)
    if model_name.lower() in {
        "flux.2-klein-4b",
        "flux.2-klein-9b",
        "flux.2-klein-9b-kv",
        "flux.2-klein-base-4b",
        "flux.2-klein-base-9b",
    }:
        return Flux2Qwen3SubfolderEmbedder(
            flux2_model_dir_or_repo=local_dir,
            device=device,
            torch_dtype=torch.bfloat16,
            load_weights=load_weights,
        )
    if not load_weights:
        raise NotImplementedError(
            f"load_weights=False is not implemented for text encoder of {model_name!r}; "
            "use flux.2-klein-* with local tokenizer + text_encoder config."
        )
    return config["text_encoder_load_fn"](device=device)


def load_ae(
    model_name: str, device: str | torch.device = "cuda", load_weights: bool = True
) -> AutoEncoder:
    config = FLUX2_MODEL_INFO[model_name.lower()]
    local_dir = _require_local_dir(model_name, check_flow_weights=load_weights)
    weight_path = os.path.join(local_dir, config["filename_ae"])

    if isinstance(device, str):
        device = torch.device(device)

    if not load_weights:
        if not os.path.exists(weight_path):
            vae_dir = os.path.join(local_dir, "vae")
            if not os.path.isdir(vae_dir):
                raise FileNotFoundError(
                    f"Missing local AE safetensors: {weight_path} and no diffusers `vae/` folder at {vae_dir}."
                )
            from diffusers import AutoencoderKLFlux2

            cfg = AutoencoderKLFlux2.load_config(local_dir, subfolder="vae")
            vae = AutoencoderKLFlux2.from_config(cfg).to(device, dtype=torch.bfloat16)
            vae.eval()
            return _DiffusersFlux2AEAdapter(vae)
        ae = AutoEncoder(AutoEncoderParams()).to(device)
        ae.eval()
        return ae

    if not os.path.exists(weight_path):
        vae_dir = os.path.join(local_dir, "vae")
        if not os.path.isdir(vae_dir):
            raise FileNotFoundError(
                f"Missing local AE safetensors: {weight_path} and no diffusers `vae/` folder at {vae_dir}."
            )

        from diffusers import AutoencoderKLFlux2

        vae = AutoencoderKLFlux2.from_pretrained(local_dir, subfolder="vae", torch_dtype=torch.bfloat16).to(device)
        vae.eval()
        return _DiffusersFlux2AEAdapter(vae)

    with torch.device("meta"):
        ae = AutoEncoder(AutoEncoderParams())

    print(f"Loading {weight_path} for the AutoEncoder weights")
    sd = load_sft(weight_path, device=str(device))
    ae.load_state_dict(sd, strict=True, assign=True)

    return ae.to(device)


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str
