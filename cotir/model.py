import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

from .adapter import CoTAdapter
from .flux.model import Flux
from .flux.modules.layers import timestep_embedding
from .flux.modules.lora import LinearLora, replace_linear_with_lora
from .flux.util import (
    configs as FLUX_MODEL_CONFIGS,
    get_checkpoint_path,
    load_sft,
    optionally_expand_state_dict,
    print_load_warning,
)
from .flux2.hf_sd_convert import maybe_convert_transformer_sd
from .flux2.model import Flux2
from .flux2.util import FLUX2_MODEL_INFO, load_flow_model


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "ema", "weights"):
            v = obj.get(k)
            if isinstance(v, dict):
                return v
        return obj
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def _strip_module_prefix(sd: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}


def _is_flux2_model(base_model_name: str) -> bool:
    return str(base_model_name).lower().startswith("flux.2")


def _load_base_state_dict(args, base_checkpoint_path: str | None, device: str) -> dict:
    base_model_name = str(args.base_model_name).lower()
    is_flux2 = _is_flux2_model(base_model_name)

    if base_checkpoint_path:
        if not os.path.exists(base_checkpoint_path):
            raise FileNotFoundError(f"Base checkpoint not found: {base_checkpoint_path}")
        if base_checkpoint_path.endswith(".safetensors"):
            if is_flux2:
                return _strip_module_prefix(
                    maybe_convert_transformer_sd(load_safetensors(base_checkpoint_path, device="cpu"))
                )
            return _strip_module_prefix(load_sft(base_checkpoint_path, device=str(device)))
        return _strip_module_prefix(_extract_state_dict(torch.load(base_checkpoint_path, map_location="cpu")))

    if is_flux2:
        if base_model_name not in FLUX2_MODEL_INFO:
            raise KeyError(f"Unsupported flux2 base model: {args.base_model_name}. Keys={list(FLUX2_MODEL_INFO.keys())}")
        print(f"Loading pretrained FLUX.2 weights: {base_model_name}")
        base = load_flow_model(base_model_name, device=device)
        sd = base.state_dict()
        del base
        return sd

    if base_model_name not in FLUX_MODEL_CONFIGS:
        raise KeyError(
            f"Unsupported kontext base model: {args.base_model_name}. Keys={list(FLUX_MODEL_CONFIGS.keys())}"
        )
    config = FLUX_MODEL_CONFIGS[base_model_name]
    ckpt_path = str(get_checkpoint_path(config.repo_id, config.repo_flow, "FLUX_MODEL"))
    print(f"Loading pretrained: {ckpt_path}")
    return _strip_module_prefix(load_sft(ckpt_path, device=str(device)))


def _merge_lora_into_base(model: nn.Module) -> int:
    merged_count = 0
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, LinearLora):
                continue
            if getattr(module, "_merged_once", False):
                continue
            a = module.lora_A.weight.to(torch.float32)
            b = module.lora_B.weight.to(torch.float32)
            module.weight.data.add_((torch.matmul(b, a) * float(module.scale)).to(module.weight.dtype))

            if module.lora_B.bias is not None:
                if module.bias is None:
                    module.bias = torch.nn.Parameter(
                        torch.zeros(module.out_features, device=module.weight.device, dtype=module.weight.dtype)
                    )
                module.bias.data.add_((module.lora_B.bias.to(torch.float32) * float(module.scale)).to(module.bias.dtype))

            module.lora_A.weight.data.zero_()
            module.lora_B.weight.data.zero_()
            if module.lora_B.bias is not None:
                module.lora_B.bias.data.zero_()
            module.scale = 0.0
            module._merged_once = True
            merged_count += 1
    return merged_count


def build_cotir_model(args) -> nn.Module:
    if _is_flux2_model(args.base_model_name):
        return CoTIRFlux2(args)
    return CoTIRFlux1(args)


def load_model(
    args,
    checkpoint_path: str | None = None,
    base_checkpoint_path: str | None = None,
    cot_adapter_checkpoint_path: str | None = None,
    lora_checkpoint_path: str | None = None,
    device: str = "cpu",
    verbose: bool = True,
    load_weights: bool = True,
    for_inference: bool = False,
    merge_lora_inference: bool | None = None,
) -> nn.Module:
    with torch.device("cpu"):
        model = build_cotir_model(args).to(torch.bfloat16)

    if getattr(args, "lora", None) and args.lora.enabled:
        print("Loading LoRA")
        model.lora_model()

    if not load_weights:
        return model

    use_split = any([base_checkpoint_path, cot_adapter_checkpoint_path, lora_checkpoint_path])
    if use_split:
        if not (base_checkpoint_path and cot_adapter_checkpoint_path and lora_checkpoint_path):
            raise ValueError(
                "Split loading requires base_checkpoint_path, cot_adapter_checkpoint_path and lora_checkpoint_path"
            )
        base_sd = _load_base_state_dict(args, base_checkpoint_path, device)
        base_sd = optionally_expand_state_dict(model, base_sd)
        model.load_state_dict(base_sd, strict=False, assign=True)

        cot_sd = _strip_module_prefix(_extract_state_dict(torch.load(cot_adapter_checkpoint_path, map_location="cpu")))
        model.load_state_dict(cot_sd, strict=False, assign=True)

        lora_sd = _strip_module_prefix(_extract_state_dict(torch.load(lora_checkpoint_path, map_location="cpu")))
        model.load_state_dict(lora_sd, strict=False, assign=True)

        do_merge = bool(merge_lora_inference)
        if merge_lora_inference is None and hasattr(args, "lora"):
            do_merge = bool(getattr(args.lora, "merge_for_inference", False))
        if for_inference and do_merge and getattr(args, "lora", None) and args.lora.enabled:
            merged_count = _merge_lora_into_base(model)
            if verbose:
                print(f"LoRA merged into base layers: {merged_count}")
        return model

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        if str(checkpoint_path).endswith(".safetensors"):
            sd = _load_base_state_dict(args, checkpoint_path, device)
        else:
            sd = _strip_module_prefix(_extract_state_dict(torch.load(checkpoint_path, map_location="cpu")))
    else:
        sd = _load_base_state_dict(args, None, device)
    sd = optionally_expand_state_dict(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if verbose:
        print_load_warning(missing, unexpected)
    return model


class CoTIRFlux2(Flux2):
    def __init__(self, args):
        params = FLUX2_MODEL_INFO[args.base_model_name.lower()]["params"]
        super().__init__(params)
        self.args = args
        self.cot_adapter = CoTAdapter(
            hidden_size=args.cot_adapter.hidden_size,
            num_heads=args.cot_adapter.num_heads,
            mlp_ratio=args.cot_adapter.mlp_ratio,
            qk_scale=args.cot_adapter.qk_scale,
            out_channels=args.cot_adapter.out_channels,
            depth=args.cot_adapter.depth,
        )

    def lora_model(self):
        replace_linear_with_lora(self.single_blocks, max_rank=self.args.lora.rank, scale=self.args.lora.scale)
        replace_linear_with_lora(self.final_layer, max_rank=self.args.lora.rank, scale=self.args.lora.scale)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        num_txt_tokens = txt.shape[1]
        timestep_emb = timestep_embedding(timesteps, 256)
        vec = self.time_in(timestep_emb)
        if getattr(self, "use_guidance_embed", False):
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + self.guidance_in(guidance_emb)

        double_block_mod_img = self.double_stream_modulation_img(vec)
        double_block_mod_txt = self.double_stream_modulation_txt(vec)
        single_block_mod, _ = self.single_stream_modulation(vec)

        img = self.img_in(img)
        txt = self.txt_in(txt)
        pe_x = self.pe_embedder(img_ids)
        pe_ctx = self.pe_embedder(txt_ids)

        for block in self.double_blocks:
            img, txt, _ = block.forward_kv_extract(
                img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, num_ref_tokens=0
            )

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        txt_out, txt_out_s, txt_out_d, txt_out_p = self.cot_adapter(txt, img, vec, pe)
        txt = txt + txt_out

        img = torch.cat((txt, img), dim=1)
        pe = torch.cat((pe_ctx, pe_x), dim=2)
        for block in self.single_blocks:
            img, _ = block.forward_kv_extract(img, pe, single_block_mod, num_txt_tokens, num_ref_tokens=0)

        img = img[:, num_txt_tokens:, ...]
        img = self.final_layer(img, vec)
        return img, txt_out_s, txt_out_d, txt_out_p


class CoTIRFlux1(Flux):
    def __init__(self, args):
        base_model_name = str(args.base_model_name).lower()
        if base_model_name not in FLUX_MODEL_CONFIGS:
            raise KeyError(
                f"Unsupported kontext base model: {args.base_model_name}. Keys={list(FLUX_MODEL_CONFIGS.keys())}"
            )
        super().__init__(FLUX_MODEL_CONFIGS[base_model_name].params)
        self.args = args
        self.cot_adapter = CoTAdapter(
            hidden_size=args.cot_adapter.hidden_size,
            num_heads=args.cot_adapter.num_heads,
            mlp_ratio=args.cot_adapter.mlp_ratio,
            qk_scale=args.cot_adapter.qk_scale,
            out_channels=args.cot_adapter.out_channels,
            depth=args.cot_adapter.depth,
        )

    def lora_model(self):
        replace_linear_with_lora(self.single_blocks, max_rank=self.args.lora.rank, scale=self.args.lora.scale)
        replace_linear_with_lora(self.final_layer, max_rank=self.args.lora.rank, scale=self.args.lora.scale)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        txt_out, txt_out_s, txt_out_d, txt_out_p = self.cot_adapter(txt, img, vec, pe)
        txt = txt + txt_out

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]
        img = self.final_layer(img, vec)
        return img, txt_out_s, txt_out_d, txt_out_p