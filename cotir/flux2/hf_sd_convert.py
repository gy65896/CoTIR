"""
Map Diffusers FLUX.2 transformer checkpoint keys to CoTIR native Flux2 keys.
"""

from __future__ import annotations

import re
from typing import Any


def _is_diffusers_flux2_transformer_sd(sd: dict[str, Any]) -> bool:
    return any(k.startswith("x_embedder.") or k.startswith("transformer_blocks.") for k in sd)


def _stack_qkv(wq: Any, wk: Any, wv: Any) -> Any:
    import torch

    return torch.cat([wq, wk, wv], dim=0)


def convert_diffusers_transformer_sd_to_cotir(sd: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    used: set[str] = set()

    def consume(key: str) -> Any:
        if key not in sd:
            raise KeyError(key)
        used.add(key)
        return sd[key]

    if "x_embedder.weight" in sd:
        out["img_in.weight"] = consume("x_embedder.weight")
    if "context_embedder.weight" in sd:
        out["txt_in.weight"] = consume("context_embedder.weight")

    te = "time_guidance_embed.timestep_embedder."
    if f"{te}linear_1.weight" in sd:
        out["time_in.in_layer.weight"] = consume(f"{te}linear_1.weight")
    if f"{te}linear_2.weight" in sd:
        out["time_in.out_layer.weight"] = consume(f"{te}linear_2.weight")

    ge1 = "time_guidance_embed.guidance_embedder.linear_1.weight"
    ge2 = "time_guidance_embed.guidance_embedder.linear_2.weight"
    if ge1 in sd and ge2 in sd:
        out["guidance_in.in_layer.weight"] = consume(ge1)
        out["guidance_in.out_layer.weight"] = consume(ge2)

    for hf_mod, cot_mod in (
        ("double_stream_modulation_img", "double_stream_modulation_img"),
        ("double_stream_modulation_txt", "double_stream_modulation_txt"),
    ):
        k = f"{hf_mod}.linear.weight"
        if k in sd:
            out[f"{cot_mod}.lin.weight"] = consume(k)

    k = "single_stream_modulation.linear.weight"
    if k in sd:
        out["single_stream_modulation.lin.weight"] = consume(k)

    tb_layers: set[int] = set()
    for key in sd:
        m = re.match(r"^transformer_blocks\.(\d+)\.", key)
        if m:
            tb_layers.add(int(m.group(1)))
    for i in sorted(tb_layers):
        p = f"transformer_blocks.{i}"
        b = f"double_blocks.{i}"
        ia, ta = f"{b}.img_attn", f"{b}.txt_attn"
        wq, wk, wv = (
            consume(f"{p}.attn.to_q.weight"),
            consume(f"{p}.attn.to_k.weight"),
            consume(f"{p}.attn.to_v.weight"),
        )
        out[f"{ia}.qkv.weight"] = _stack_qkv(wq, wk, wv)
        out[f"{ia}.norm.query_norm.scale"] = consume(f"{p}.attn.norm_q.weight")
        out[f"{ia}.norm.key_norm.scale"] = consume(f"{p}.attn.norm_k.weight")
        out[f"{ia}.proj.weight"] = consume(f"{p}.attn.to_out.0.weight")
        wq, wk, wv = (
            consume(f"{p}.attn.add_q_proj.weight"),
            consume(f"{p}.attn.add_k_proj.weight"),
            consume(f"{p}.attn.add_v_proj.weight"),
        )
        out[f"{ta}.qkv.weight"] = _stack_qkv(wq, wk, wv)
        out[f"{ta}.norm.query_norm.scale"] = consume(f"{p}.attn.norm_added_q.weight")
        out[f"{ta}.norm.key_norm.scale"] = consume(f"{p}.attn.norm_added_k.weight")
        out[f"{ta}.proj.weight"] = consume(f"{p}.attn.to_add_out.weight")
        out[f"{b}.img_mlp.0.weight"] = consume(f"{p}.ff.linear_in.weight")
        out[f"{b}.img_mlp.2.weight"] = consume(f"{p}.ff.linear_out.weight")
        out[f"{b}.txt_mlp.0.weight"] = consume(f"{p}.ff_context.linear_in.weight")
        out[f"{b}.txt_mlp.2.weight"] = consume(f"{p}.ff_context.linear_out.weight")

    sb_layers: set[int] = set()
    for key in sd:
        m = re.match(r"^single_transformer_blocks\.(\d+)\.", key)
        if m:
            sb_layers.add(int(m.group(1)))
    for i in sorted(sb_layers):
        p = f"single_transformer_blocks.{i}"
        b = f"single_blocks.{i}"
        out[f"{b}.linear1.weight"] = consume(f"{p}.attn.to_qkv_mlp_proj.weight")
        out[f"{b}.linear2.weight"] = consume(f"{p}.attn.to_out.weight")
        out[f"{b}.norm.query_norm.scale"] = consume(f"{p}.attn.norm_q.weight")
        out[f"{b}.norm.key_norm.scale"] = consume(f"{p}.attn.norm_k.weight")

    if "norm_out.linear.weight" in sd:
        out["final_layer.adaLN_modulation.1.weight"] = consume("norm_out.linear.weight")
    if "proj_out.weight" in sd:
        out["final_layer.linear.weight"] = consume("proj_out.weight")

    expected = {k for k in sd.keys() if k != "__metadata__"}
    leftover = expected - used
    if leftover:
        raise RuntimeError(
            "Unmapped Diffusers keys: "
            + ", ".join(sorted(leftover)[:16])
            + (f" ... (+{len(leftover) - 16} more)" if len(leftover) > 16 else "")
        )
    return out


def maybe_convert_transformer_sd(sd: dict[str, Any]) -> dict[str, Any]:
    if _is_diffusers_flux2_transformer_sd(sd):
        return convert_diffusers_transformer_sd_to_cotir(sd)
    return sd
