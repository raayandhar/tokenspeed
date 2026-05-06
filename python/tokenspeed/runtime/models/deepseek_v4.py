# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Inference-only DeepSeek V4 model skeleton.

This module intentionally registers only architecture pieces that map to the
DeepSeek V4 Flash checkpoint. The sparse MLA forward path still fails loudly
until the HCA/CSA cache kernels are wired into TokenSpeed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_hca_compress_kv_cache_insert,
    deepseek_v4_indexer_topk_reference,
    deepseek_v4_inv_rope_reference,
    deepseek_v4_prepare_indexer_q_reference,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
    read_deepseek_v4_indexer_fp8_cache,
    read_deepseek_v4_indexer_mxfp4_cache,
    save_deepseek_v4_compressor_state,
)
from tokenspeed.runtime.layers.layernorm import RMSNorm
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.moe.checkpoint import (
    ExpertCheckpointSchema,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.layer import MoELayer
from tokenspeed.runtime.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
)
from tokenspeed.runtime.layers.moe.utils import RoutingMethodType
from tokenspeed.runtime.layers.quantization import Mxfp4Config
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.env import global_server_args_dict


def _dequant_fp8_weight(layer: nn.Module, shape: tuple[int, ...]) -> torch.Tensor:
    weight = layer.weight.view(*shape)
    scale = getattr(layer, "weight_scale_inv", None)
    if scale is None or weight.dtype != torch.float8_e4m3fn:
        return weight.float()

    cache = getattr(layer, "_deepseek_v4_dequant_cache", None)
    if cache is not None:
        cached_shape, cached_weight = cache
        if cached_shape == tuple(shape):
            return cached_weight

    block_n, block_k = getattr(layer.quant_config, "weight_block_size", (128, 128))
    if len(shape) == 2:
        out_dim, in_dim = shape
        scale = scale.view(
            (out_dim + block_n - 1) // block_n,
            (in_dim + block_k - 1) // block_k,
        )
        expanded_scale = (
            scale.float()
            .repeat_interleave(block_n, dim=0)
            .repeat_interleave(block_k, dim=1)
        )
        out = weight.float() * expanded_scale[:out_dim, :in_dim]
        layer._deepseek_v4_dequant_cache = (tuple(shape), out)
        return out

    groups, out_dim, in_dim = shape
    scale = scale.view(
        groups,
        (out_dim + block_n - 1) // block_n,
        (in_dim + block_k - 1) // block_k,
    )
    expanded_scale = (
        scale.float()
        .repeat_interleave(block_n, dim=1)
        .repeat_interleave(block_k, dim=2)
    )
    out = weight.float() * expanded_scale[:, :out_dim, :in_dim]
    layer._deepseek_v4_dequant_cache = (tuple(shape), out)
    return out


def _fp8_act_quant_dequant(x: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """Simulate DeepSeek V4's block FP8 activation quantization."""

    if x.shape[-1] % block_size != 0:
        raise ValueError(
            f"DeepSeek V4 FP8 activation quantization expects K divisible by "
            f"{block_size}, got {x.shape[-1]}"
        )
    orig_shape = x.shape
    x_blocks = x.float().reshape(-1, orig_shape[-1]).unflatten(-1, (-1, block_size))
    amax = x_blocks.abs().amax(dim=-1).clamp_min(1.0e-4)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    scale = scale.to(torch.float8_e8m0fnu).float()
    quantized = (
        (x_blocks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    )
    return (quantized.float() * scale.unsqueeze(-1)).flatten(-2).reshape(orig_shape)


def _fp8_linear(
    layer: nn.Module,
    x: torch.Tensor,
    shape: tuple[int, ...],
    *,
    quantize_act: bool = True,
) -> torch.Tensor:
    weight = _dequant_fp8_weight(layer, shape)
    x_eff = (
        _fp8_act_quant_dequant(x, DEEPSEEK_V4_FP8_BLOCK_SIZE)
        if quantize_act and layer.weight.dtype == torch.float8_e4m3fn
        else x.float()
    )
    return torch.matmul(x_eff, weight.transpose(-2, -1)).to(x.dtype)


def _deepseek_v4_reorder_c4_ape_2604(ape: torch.Tensor) -> torch.Tensor:
    """Convert C4 overlap APE from checkpoint layout to runtime window layout."""

    if ape.dim() != 2 or ape.shape[0] != 4 or ape.shape[1] % 2 != 0:
        raise ValueError(f"expected C4 APE [4, even], got {tuple(ape.shape)}")
    older, newer = ape.chunk(2, dim=-1)
    return torch.cat([older, newer], dim=0).reshape_as(ape)


def _sinkhorn(mixes: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    if iters < 1:
        raise ValueError(f"sinkhorn iterations must be >= 1, got {iters}")
    mixes = torch.softmax(mixes, dim=-1) + eps
    mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        mixes = mixes / (mixes.sum(dim=-1, keepdim=True) + eps)
        mixes = mixes / (mixes.sum(dim=-2, keepdim=True) + eps)
    return mixes


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch hidden-compression pre step.

    Shapes follow the DeepSeek V4 hidden-compression contract:
    residual [T, M, H], layer input [T, H], post [T, M, 1], comb [T, M, M].
    """

    if residual.dim() != 3:
        raise ValueError(f"expected residual [T, M, H], got {tuple(residual.shape)}")
    num_tokens, hc_mult, hidden_size = residual.shape
    x = residual.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x, fn.float()) * rsqrt
    expected = (2 + hc_mult) * hc_mult
    if mixes.shape[-1] != expected:
        raise ValueError(f"expected {expected} HC mixes, got {mixes.shape[-1]}")

    pre_raw, post_raw, comb_raw = torch.split(
        mixes, [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre_base, post_base, comb_base = torch.split(
        hc_base.float(), [hc_mult, hc_mult, hc_mult * hc_mult], dim=-1
    )
    pre = torch.sigmoid(pre_raw * hc_scale[0].float() + pre_base) + hc_eps
    post = (torch.sigmoid(post_raw * hc_scale[1].float() + post_base) * 2.0).unsqueeze(
        -1
    )
    comb_logits = comb_raw.reshape(num_tokens, hc_mult, hc_mult)
    comb_base = comb_base.reshape(1, hc_mult, hc_mult)
    comb = _sinkhorn(
        comb_logits * hc_scale[2].float() + comb_base,
        iters=sinkhorn_iters,
        eps=hc_eps,
    )
    layer_input = torch.sum(pre.unsqueeze(-1) * residual.float(), dim=1)
    return layer_input.to(residual.dtype), post, comb


def mhc_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    if post.dim() == 2:
        post = post.unsqueeze(-1)
    mixed_residual = torch.einsum("tnm,tnh->tmh", comb.float(), residual.float())
    block_update = post.float() * hidden_states.float().unsqueeze(1)
    return (mixed_residual + block_update).to(hidden_states.dtype)


def hc_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    shape, dtype = hidden_states.size(), hidden_states.dtype
    x = hidden_states.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = F.linear(x, hc_fn.float()) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)


def deepseek_v4_select_experts(
    router_logits: torch.Tensor,
    top_k: int,
    renormalize: bool,
    *,
    correction_bias: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DeepSeek V4 MoE routing.

    DeepSeek V4 uses sqrt(softplus(logits)) as expert scores. Correction bias
    only affects expert selection; the gathered expert weights come from the
    unbiased scores. Hash-routed layers use checkpoint-provided expert ids but
    still gather weights from the gate scores.
    """

    scores = torch.sqrt(F.softplus(router_logits.float()))
    if hash_indices_table is not None:
        if input_ids is None:
            raise ValueError("hash-routed DeepSeek V4 MoE requires input_ids")
        topk_ids = hash_indices_table[input_ids.reshape(-1)].to(
            device=scores.device,
            dtype=torch.long,
        )
    else:
        scores_for_choice = scores
        if correction_bias is not None:
            scores_for_choice = scores_for_choice + correction_bias.to(
                device=scores.device,
                dtype=scores.dtype,
            ).unsqueeze(0)
        topk_ids = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=True)[1]

    topk_weights = scores.gather(1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(topk_weights.dtype).tiny
        )
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32), scores


def pack_topk_as_router_logits(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Encode preselected top-k weights for BYPASSED TokenSpeed MoE backends.

    MXFP4 backends currently build routing data from logits internally. Packing
    the normalized top-k weights as log-probabilities with very negative values
    elsewhere makes their TopK -> Softmax/Renormalize route recover the same
    selected ids and weights without changing the shared backend.
    """

    router_logits = torch.full(
        (topk_ids.shape[0], num_experts),
        -1e20,
        dtype=torch.float32,
        device=topk_weights.device,
    )
    safe_weights = topk_weights.clamp_min(torch.finfo(torch.float32).tiny)
    router_logits.scatter_(1, topk_ids.long(), safe_weights.log())
    return router_logits


DEEPSEEK_V4_COMPRESSED_CACHE_ALIGNMENT = 576
DEEPSEEK_V4_FP8_BLOCK_SIZE = 128
DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32


@dataclass(frozen=True)
class DeepseekV4AttentionLayout:
    """Static DeepSeek V4 sparse MLA contract.

    `swa_head_bytes` is the uint8 SWA cache row width used by DeepSeek V4:
    FP8 NoPE bytes, BF16 RoPE bytes, UE8M0 scale bytes, then one pad byte.
    """

    kind: str
    compress_ratio: int
    num_heads: int
    num_local_heads: int
    padded_heads: int
    head_dim: int
    nope_head_dim: int
    rope_head_dim: int
    swa_window: int
    swa_head_bytes: int
    compressed_cache_alignment: int
    needs_compressed_cache: bool
    needs_indexer: bool
    indexer_cache_head_bytes: int | None = None


def _deepseek_v4_padded_heads(num_local_heads: int) -> int:
    if num_local_heads <= 64:
        return 64
    if num_local_heads <= 128:
        return 128
    raise ValueError(
        f"DeepSeek V4 attention supports at most 128 local heads, "
        f"got {num_local_heads}"
    )


def _attention_use_fp4_indexer_cache(config: PretrainedConfig) -> bool:
    override = global_server_args_dict.get("attention_use_fp4_indexer_cache", None)
    if override is not None:
        return bool(override)
    attention_config = getattr(config, "attention_config", None)
    if isinstance(attention_config, dict):
        return bool(attention_config.get("use_fp4_indexer_cache", False))
    return bool(getattr(attention_config, "use_fp4_indexer_cache", False))


def deepseek_v4_rope_config(
    config: PretrainedConfig, compress_ratio: int
) -> tuple[float, dict | None]:
    """Return the per-layer DeepSeek V4 RoPE base and scaling config.

    DeepSeek V4 uses ordinary RoPE for SWA-only layers. Compressed layers
    use the checkpoint's separate `compress_rope_theta` together with YaRN.
    """

    if compress_ratio <= 1:
        return float(getattr(config, "rope_theta", 10000.0)), None

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        rope_scaling["rope_type"] = "deepseek_yarn"
        rope_scaling["mscale"] = 0
        rope_scaling["mscale_all_dim"] = 0
    return (
        float(
            getattr(
                config,
                "compress_rope_theta",
                getattr(config, "rope_theta", 10000.0),
            )
        ),
        rope_scaling,
    )


def deepseek_v4_attention_layout(
    config: PretrainedConfig,
    layer_index: int,
    attn_tp_size: int = 1,
    use_fp4_indexer_cache: bool = False,
) -> DeepseekV4AttentionLayout:
    """Return the per-layer V4 attention/cache layout before kernel wiring.

    This keeps TokenSpeed's model code aligned with the three DeepSeek V4
    attention cases: SWA-only (`compress_ratio <= 1`), HCA (`128`), and CSA
    (`4` with indexer).
    """

    compress_ratio = max(1, int(config.compress_ratios[layer_index]))
    if compress_ratio <= 1:
        kind = "swa"
    elif compress_ratio == 4:
        kind = "csa"
    elif compress_ratio == 128:
        kind = "hca"
    else:
        raise ValueError(
            f"Unsupported DeepSeek V4 compress_ratio={compress_ratio}; "
            "expected 1, 4, or 128."
        )

    if config.num_attention_heads % attn_tp_size != 0:
        raise ValueError(
            f"num_attention_heads={config.num_attention_heads} must be divisible "
            f"by attn_tp_size={attn_tp_size}"
        )
    num_local_heads = config.num_attention_heads // attn_tp_size
    head_dim = int(config.head_dim)
    rope_head_dim = int(config.qk_rope_head_dim)
    nope_head_dim = head_dim - rope_head_dim
    if nope_head_dim <= 0:
        raise ValueError(
            f"head_dim={head_dim} must be larger than rope_head_dim={rope_head_dim}"
        )
    if nope_head_dim % 64 != 0:
        raise ValueError(
            f"DeepSeek V4 FP8 NoPE dim must be divisible by 64, got {nope_head_dim}"
        )

    swa_head_bytes = nope_head_dim + rope_head_dim * 2 + nope_head_dim // 64 + 1
    indexer_cache_head_bytes = None
    if kind == "csa":
        index_head_dim = int(config.index_head_dim)
        if use_fp4_indexer_cache:
            indexer_cache_head_bytes = (
                index_head_dim // 2 + index_head_dim // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
            )
        else:
            indexer_cache_head_bytes = (
                index_head_dim + (index_head_dim // DEEPSEEK_V4_FP8_BLOCK_SIZE) * 4
            )

    return DeepseekV4AttentionLayout(
        kind=kind,
        compress_ratio=compress_ratio,
        num_heads=int(config.num_attention_heads),
        num_local_heads=num_local_heads,
        padded_heads=_deepseek_v4_padded_heads(num_local_heads),
        head_dim=head_dim,
        nope_head_dim=nope_head_dim,
        rope_head_dim=rope_head_dim,
        swa_window=int(getattr(config, "sliding_window", 128)),
        swa_head_bytes=swa_head_bytes,
        compressed_cache_alignment=DEEPSEEK_V4_COMPRESSED_CACHE_ALIGNMENT,
        needs_compressed_cache=compress_ratio > 1,
        needs_indexer=kind == "csa",
        indexer_cache_head_bytes=indexer_cache_head_bytes,
    )


class DeepseekV4MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        is_shared_expert: bool = False,
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        tp = mapping.moe if is_shared_expert else mapping.dense
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            tp_rank=tp.tp_ep_rank if is_shared_expert else tp.tp_rank,
            tp_size=tp.tp_ep_size if is_shared_expert else tp.tp_size,
            tp_group=tp.tp_ep_group if is_shared_expert else tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,
            tp_rank=tp.tp_ep_rank if is_shared_expert else tp.tp_rank,
            tp_size=tp.tp_ep_size if is_shared_expert else tp.tp_size,
            tp_group=tp.tp_ep_group if is_shared_expert else tp.tp_group,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] == 0:
            return x.new_empty((0, self.down_proj.output_size))
        gate_up = _fp8_linear(
            self.gate_up_proj,
            x,
            (
                self.gate_up_proj.output_size_per_partition,
                self.gate_up_proj.input_size,
            ),
        )
        gate, up = gate_up.float().chunk(2, dim=-1)
        if self.swiglu_limit is not None and self.swiglu_limit > 0:
            gate = torch.clamp(gate, max=self.swiglu_limit)
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        x = (F.silu(gate) * up).to(x.dtype)
        return _fp8_linear(
            self.down_proj,
            x,
            (self.down_proj.output_size, self.down_proj.input_size_per_partition),
        )


class DeepseekV4MoEGate(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_index: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.hidden_size)
        )
        self.is_hash_moe = layer_index < config.num_hash_layers
        if self.is_hash_moe:
            self.tid2eid = nn.Parameter(
                torch.empty(
                    config.vocab_size, config.num_experts_per_tok, dtype=torch.int32
                ),
                requires_grad=False,
            )
            self.e_score_correction_bias = None
        elif getattr(config, "topk_method", None) == "noaux_tc":
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )
        else:
            self.register_parameter("tid2eid", None)
            self.e_score_correction_bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.weight, None)


class DeepseekV4MoE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_index: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.layer_index = layer_index
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.gate = DeepseekV4MoEGate(config, layer_index)
        self.scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(
                f"Unsupported DeepSeek V4 MoE scoring: {self.scoring_func}"
            )

        if config.n_shared_experts is not None:
            self.shared_experts = DeepseekV4MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
                mapping,
                quant_config,
                add_prefix("shared_experts", prefix),
                is_shared_expert=True,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )
        else:
            self.shared_experts = None

        routed_quant_config = Mxfp4Config(
            ignored_layers=getattr(quant_config, "ignored_layers", None),
            is_checkpoint_mxfp4_serialized=True,
        )
        self.experts = MoELayer(
            top_k=config.num_experts_per_tok,
            num_experts=config.n_routed_experts
            + global_server_args_dict["ep_num_redundant_experts"],
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=routed_quant_config,
            layer_index=layer_index,
            prefix=prefix,
            tp_rank=mapping.moe.tp_rank,
            tp_size=mapping.moe.tp_size,
            ep_rank=mapping.moe.ep_rank,
            ep_size=mapping.moe.ep_size,
            activation="swiglu",
            swiglu_limit=getattr(config, "swiglu_limit", None),
            with_bias=True,
            routing_config={
                "routed_scaling_factor": self.routed_scaling_factor,
                "correction_bias": self.gate.e_score_correction_bias,
                "routing_method_type": RoutingMethodType.Renormalize,
            },
        )
        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            correction_bias=self.gate.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                self.experts.apply_routed_scaling_factor_on_output
            ),
            output_format=self.experts.topk_output_format,
        )

    def _select_experts(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)
        return deepseek_v4_select_experts(
            router_logits,
            self.config.num_experts_per_tok,
            self.config.norm_topk_prob,
            correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=self.gate.tid2eid,
            input_ids=input_ids,
        )

    def _make_topk_output(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_scores: torch.Tensor,
    ):
        if self.experts.topk_output_format.is_bypassed():
            router_logits = pack_topk_as_router_logits(
                topk_weights, topk_ids, self.config.n_routed_experts
            )
            return BypassedTopKOutput(
                hidden_states, router_logits, self.topk.topk_config
            )
        return StandardTopKOutput(topk_weights, topk_ids, router_scores)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        topk_weights, topk_ids, router_scores = self._select_experts(
            hidden_states, input_ids
        )
        topk_output = self._make_topk_output(
            hidden_states, topk_weights, topk_ids, router_scores
        )
        routed = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
        )
        if self.routed_scaling_factor != 1.0:
            routed *= self.routed_scaling_factor
        if self.shared_experts is not None:
            shared = self.shared_experts(hidden_states)
            routed = routed + shared
        return routed


class DeepseekV4Compressor(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        head_dim: int,
        compress_ratio: int,
        prefix: str,
    ) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        self.coff = 2 if self.overlap else 1
        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(compress_ratio, self.coff * head_dim, dtype=state_dtype),
            requires_grad=False,
        )
        self._ape_reordered = False
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            hidden_size,
            [self.coff * head_dim, self.coff * head_dim],
            bias=False,
            quant_config=None,
            prefix=add_prefix("fused_wkv_wgate", prefix),
        )
        self.norm = RMSNorm(head_dim, eps=config.rms_norm_eps)

    def process_weights_after_loading(self, module=None) -> None:
        del module
        if not self.overlap or self._ape_reordered:
            return
        with torch.no_grad():
            self.ape.data.copy_(_deepseek_v4_reorder_c4_ape_2604(self.ape.data))
        self._ape_reordered = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
        *,
        state_cache: torch.Tensor | None = None,
        write_compressed_cache: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 compressor requires forward metadata")
        weight = _dequant_fp8_weight(
            self.fused_wkv_wgate,
            (
                self.fused_wkv_wgate.output_size_per_partition,
                self.fused_wkv_wgate.input_size,
            ),
        )
        kv_score = torch.matmul(hidden_states.float(), weight.transpose(0, 1))
        kv, score = kv_score.split([self.coff * self.head_dim] * 2, dim=-1)
        if state_cache is None:
            state_cache = pool.get_compressor_state_buffer(layer_index)
        save_deepseek_v4_compressor_state(
            kv=kv,
            score=score,
            ape=self.ape,
            state_cache=state_cache,
            slot_mapping=out_cache_loc,
            positions=positions,
            block_size=pool.state_block_size,
            compress_ratio=self.compress_ratio,
        )
        if not write_compressed_cache:
            return kv, score

        compressed_slots = metadata.compressed_slot_mapping(
            positions, self.compress_ratio
        )
        insert = (
            deepseek_v4_csa_compress_kv_cache_insert
            if self.compress_ratio == 4
            else deepseek_v4_hca_compress_kv_cache_insert
        )
        insert(
            state_cache=state_cache,
            token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
            positions=positions,
            compressor_slot_mapping=out_cache_loc,
            block_table=metadata.block_table,
            compressor_block_size=pool.state_block_size,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.norm.variance_epsilon,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=pool.get_compressed_kv_buffer_2d(layer_index),
            kv_slot_mapping=compressed_slots,
            kv_cache_block_size=pool.compressed_block_size,
            compress_ratio=self.compress_ratio,
        )
        return kv, score


class DeepseekV4Indexer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
        compress_ratio: int,
    ) -> None:
        super().__init__()
        self.wq_b = ReplicatedLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            config.hidden_size,
            config.index_n_heads,
            bias=False,
            quant_config=None,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = DeepseekV4Compressor(
            config,
            config.hidden_size,
            config.index_head_dim,
            compress_ratio,
            add_prefix("compressor", prefix),
        )
        self.use_fp4_cache = _attention_use_fp4_indexer_cache(config)
        self.compress_ratio = compress_ratio
        self.n_head = int(config.index_n_heads)
        self.head_dim = int(config.index_head_dim)
        self.topk_tokens = int(config.index_topk)
        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        qr: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        layer_index: int,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        pool = ctx.token_to_kv_pool
        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 indexer requires forward metadata")
        indexer_state = pool.get_indexer_state_buffer(layer_index)
        self.compressor(
            hidden_states=hidden_states,
            positions=positions,
            ctx=ctx,
            out_cache_loc=out_cache_loc,
            layer_index=layer_index,
            cos_sin_cache=cos_sin_cache,
            state_cache=indexer_state,
            write_compressed_cache=False,
        )
        compressed_slots = metadata.compressed_slot_mapping(
            positions, self.compress_ratio
        )
        deepseek_v4_csa_indexer_cache_insert(
            state_cache=indexer_state,
            token_to_req_indices=metadata.token_to_req_indices[: positions.numel()],
            positions=positions,
            compressor_slot_mapping=out_cache_loc,
            block_table=metadata.block_table,
            compressor_block_size=pool.state_block_size,
            rms_norm_weight=self.compressor.norm.weight,
            rms_norm_eps=self.compressor.norm.variance_epsilon,
            cos_sin_cache=cos_sin_cache,
            kv_cache_2d=pool.get_indexer_kv_buffer_2d(layer_index),
            kv_slot_mapping=compressed_slots,
            kv_cache_block_size=pool.compressed_block_size,
            use_fp4_cache=self.use_fp4_cache,
            compress_ratio=self.compress_ratio,
        )
        index_q, _ = self.wq_b(qr)
        index_q = index_q.view(-1, self.n_head, self.head_dim)
        weights, _ = self.weights_proj(hidden_states)
        index_q, weights = deepseek_v4_prepare_indexer_q_reference(
            index_q=index_q,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            weights=weights,
            softmax_scale=self.softmax_scale,
            head_scale=self.n_head**-0.5,
            use_fp4=self.use_fp4_cache,
        )
        cache_reader = (
            read_deepseek_v4_indexer_mxfp4_cache
            if self.use_fp4_cache
            else read_deepseek_v4_indexer_fp8_cache
        )
        topk = torch.full(
            (positions.numel(), self.topk_tokens),
            -1,
            device=positions.device,
            dtype=torch.int32,
        )
        for token_idx in range(positions.numel()):
            position = int(positions[token_idx].item())
            num_compressed = (position + 1) // self.compress_ratio
            if num_compressed <= 0:
                continue
            req_idx = int(metadata.token_to_req_indices[token_idx].item())
            local = torch.arange(
                num_compressed, device=positions.device, dtype=torch.int64
            )
            pages = torch.div(local, pool.compressed_block_size, rounding_mode="floor")
            offsets = local % pool.compressed_block_size
            page_ids = metadata.block_table[req_idx, pages.long()].to(torch.int64)
            slots = page_ids * pool.compressed_block_size + offsets
            k = cache_reader(
                pool.get_indexer_kv_buffer_2d(layer_index),
                slots,
                block_size=pool.compressed_block_size,
            )
            selected = min(num_compressed, self.topk_tokens)
            token_topk = deepseek_v4_indexer_topk_reference(
                index_q[token_idx : token_idx + 1],
                k,
                weights[token_idx : token_idx + 1],
                top_k=selected,
            )[0]
            topk[token_idx, :selected] = token_topk
        return topk


class DeepseekV4Attention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        layer_index: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        use_fp4_indexer_cache = _attention_use_fp4_indexer_cache(config)
        self.layout = deepseek_v4_attention_layout(
            config,
            layer_index,
            attn_tp_size=mapping.attn.tp_size,
            use_fp4_indexer_cache=use_fp4_indexer_cache,
        )
        self.compress_ratio = self.layout.compress_ratio
        self.attention_kind = self.layout.kind
        self.num_heads = self.layout.num_heads
        self.num_local_heads = self.layout.num_local_heads
        self.padded_heads = self.layout.padded_heads
        self.head_dim = self.layout.head_dim
        self.qk_rope_head_dim = self.layout.rope_head_dim
        self.nope_head_dim = self.layout.nope_head_dim
        self.scale = self.head_dim**-0.5
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.o_groups = config.o_groups
        self.num_local_groups = self.o_groups // mapping.attn.tp_size
        self.attn_sink = nn.Parameter(
            torch.full((self.padded_heads,), -float("inf"), dtype=torch.float32),
            requires_grad=False,
        )
        rope_base, rope_scaling = deepseek_v4_rope_config(config, self.compress_ratio)
        self.rotary_emb = get_rope(
            self.qk_rope_head_dim,
            rotary_dim=self.qk_rope_head_dim,
            max_position=getattr(config, "max_position_embeddings", 8192),
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
        )
        if not rope_scaling and hasattr(self.rotary_emb, "forward_cuda"):
            self.rotary_emb.forward = self.rotary_emb.forward_cuda
        self.indexer_rotary_emb = self.rotary_emb
        self.fused_wqa_wkv = MergedColumnParallelLinear(
            config.hidden_size,
            [self.q_lora_rank, self.head_dim],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("fused_wqa_wkv", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.wo_a = ColumnParallelLinear(
            self.num_heads * self.head_dim // self.o_groups,
            self.o_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.num_local_groups
        self.wo_b = RowParallelLinear(
            self.o_groups * self.o_lora_rank,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
        )
        if self.compress_ratio > 1:
            self.compressor = DeepseekV4Compressor(
                config,
                config.hidden_size,
                self.head_dim,
                self.compress_ratio,
                add_prefix("compressor", prefix),
            )
        else:
            self.compressor = None
        if self.compress_ratio == 4:
            self.indexer = DeepseekV4Indexer(
                config,
                mapping,
                quant_config,
                add_prefix("indexer", prefix),
                self.compress_ratio,
            )
        else:
            self.indexer = None

    def _split_qr_kv(self, qr_kv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)

    def _project_q_kv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qr_kv = _fp8_linear(
            self.fused_wqa_wkv,
            hidden_states,
            (
                self.fused_wqa_wkv.output_size_per_partition,
                self.fused_wqa_wkv.input_size,
            ),
        )
        qr, kv = self._split_qr_kv(qr_kv)
        qr = self.q_norm(qr)
        kv = self.kv_norm(kv.contiguous())
        q = _fp8_linear(
            self.wq_b,
            qr,
            (self.wq_b.output_size_per_partition, self.wq_b.input_size),
        )
        q = q.view(-1, self.num_local_heads, self.head_dim)
        return q, kv, qr

    def _make_padded_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.empty(
            (hidden_states.shape[0], self.padded_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

    def _cos_sin_cache(self) -> torch.Tensor:
        cache = self.rotary_emb.cos_sin_cache
        return cache if cache.dtype == torch.float32 else cache.float()

    def _insert_swa_cache(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        swa_kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> None:
        if q.shape[0] == 0:
            return
        fused_qnorm_rope_kv_insert(
            q=q,
            kv=kv,
            swa_kv_cache_2d=swa_kv_cache.view(swa_kv_cache.shape[0], -1),
            slot_mapping=slot_mapping,
            positions=positions,
            cos_sin_cache=self._cos_sin_cache(),
            rms_norm_eps=self.q_norm.variance_epsilon,
            block_size=block_size,
        )

    def _slots_from_local_indices(
        self,
        metadata,
        req_idx: int,
        local_indices: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if local_indices.numel() == 0:
            return torch.empty(0, device=local_indices.device, dtype=torch.int64)
        pages = torch.div(local_indices, block_size, rounding_mode="floor")
        offsets = local_indices % block_size
        page_ids = metadata.block_table[req_idx, pages.long()].to(torch.int64)
        return page_ids * block_size + offsets

    def _swa_slots_for_token(
        self,
        metadata,
        token_idx: int,
        position: int,
        block_size: int,
    ) -> torch.Tensor:
        start = max(0, position - self.layout.swa_window + 1)
        local = torch.arange(
            start,
            position + 1,
            device=metadata.block_table.device,
            dtype=torch.int64,
        )
        req_idx = int(metadata.token_to_req_indices[token_idx].item())
        return self._slots_from_local_indices(metadata, req_idx, local, block_size)

    def _compressed_slots_for_token(
        self,
        metadata,
        token_idx: int,
        position: int,
        block_size: int,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.compress_ratio <= 1:
            return torch.empty(0, device=metadata.block_table.device, dtype=torch.int64)
        if self.compress_ratio == 4:
            if topk_indices is None:
                raise RuntimeError("CSA attention requires indexer top-k indices")
            local = topk_indices[token_idx].to(torch.int64)
            local = local[local >= 0]
        else:
            num_compressed = (position + 1) // self.compress_ratio
            local = torch.arange(
                num_compressed,
                device=metadata.block_table.device,
                dtype=torch.int64,
            )
        req_idx = int(metadata.token_to_req_indices[token_idx].item())
        return self._slots_from_local_indices(metadata, req_idx, local, block_size)

    def _forward_flashmla_sparse(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        topk_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        try:
            from flash_mla import (
                flash_mla_sparse_fwd,
            )
        except Exception as exc:
            raise DeepseekV4AttentionOpUnavailable(
                "DeepSeek V4 requires FlashMLA sparse attention. Build/install "
                "`tokenspeed-kernel/python` with FlashMLA before serving V4."
            ) from exc

        metadata = ctx.attn_backend.forward_metadata
        if metadata is None:
            raise RuntimeError("DeepSeek V4 attention requires forward metadata")
        pool = ctx.token_to_kv_pool
        kernel_heads = self.padded_heads
        q_padded = torch.zeros(
            (q.shape[0], kernel_heads, self.head_dim),
            device=q.device,
            dtype=q.dtype,
        )
        q_padded[:, : self.num_local_heads].copy_(q)

        per_token_slots: list[tuple[torch.Tensor, torch.Tensor]] = []
        max_candidates = 0
        for token_idx in range(positions.numel()):
            position = int(positions[token_idx].item())
            compressed = self._compressed_slots_for_token(
                metadata,
                token_idx,
                position,
                pool.compressed_block_size,
                topk_indices,
            )
            swa = self._swa_slots_for_token(
                metadata, token_idx, position, pool.swa_block_size
            )
            per_token_slots.append((compressed, swa))
            max_candidates = max(max_candidates, compressed.numel() + swa.numel())
        padded_topk = max(128, ((max_candidates + 127) // 128) * 128)
        indices = torch.full(
            (positions.numel(), padded_topk),
            -1,
            device=q.device,
            dtype=torch.int32,
        )
        lengths = torch.zeros(positions.numel(), device=q.device, dtype=torch.int32)
        rows = []
        cursor = 0
        compressed_cache = (
            pool.get_compressed_kv_buffer_2d(self.layer_index)
            if self.compress_ratio > 1
            else None
        )
        swa_cache = pool.get_swa_kv_buffer(self.layer_index)
        for token_idx, (compressed, swa) in enumerate(per_token_slots):
            token_rows = []
            if compressed.numel() > 0:
                assert compressed_cache is not None
                token_rows.append(
                    dequantize_deepseek_v4_fp8_ds_mla_cache(
                        compressed_cache,
                        compressed,
                        block_size=pool.compressed_block_size,
                    )
                )
            if swa.numel() > 0:
                token_rows.append(
                    dequantize_deepseek_v4_fp8_ds_mla_cache(
                        swa_cache,
                        swa,
                        block_size=pool.swa_block_size,
                    )
                )
            if token_rows:
                joined = torch.cat(token_rows, dim=0)
                rows.append(joined)
                count = joined.shape[0]
                indices[token_idx, :count] = torch.arange(
                    cursor, cursor + count, device=q.device, dtype=torch.int32
                )
                lengths[token_idx] = count
                cursor += count
        if rows:
            kv = torch.cat(rows, dim=0)
        else:
            kv = torch.zeros(1, self.head_dim, device=q.device, dtype=torch.bfloat16)

        out, _, _ = flash_mla_sparse_fwd(
            q=q_padded,
            kv=kv.view(-1, 1, self.head_dim),
            indices=indices.unsqueeze(1),
            sm_scale=self.scale,
            attn_sink=self.attn_sink,
            topk_length=lengths,
        )
        return out[:, : self.num_local_heads]

    def _dequant_fp8_weight(
        self, layer: nn.Module, shape: tuple[int, ...]
    ) -> torch.Tensor:
        return _dequant_fp8_weight(layer, shape)

    def _fp8_linear(self, layer: nn.Module, x: torch.Tensor, shape: tuple[int, ...]):
        return _fp8_linear(layer, x, shape)

    def _dequant_wo_a_weight(self) -> torch.Tensor:
        in_dim = self.num_heads * self.head_dim // self.o_groups
        return self._dequant_fp8_weight(
            self.wo_a, (self.num_local_groups, self.o_lora_rank, in_dim)
        )

    def _project_attention_output(
        self,
        attn_output: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        heads_per_group = self.num_local_heads // self.num_local_groups
        grouped = deepseek_v4_inv_rope_reference(
            attn_output,
            positions,
            self._cos_sin_cache(),
            n_groups=self.num_local_groups,
            heads_per_group=heads_per_group,
            nope_dim=self.nope_head_dim,
            rope_dim=self.qk_rope_head_dim,
        )
        weight = self._dequant_wo_a_weight()
        z = torch.bmm(
            grouped.float().transpose(0, 1),
            weight.transpose(1, 2),
        ).transpose(0, 1)
        z = z.to(attn_output.dtype).contiguous()
        out = self._fp8_linear(
            self.wo_b,
            z.flatten(1),
            (self.wo_b.output_size, self.wo_b.input_size_per_partition),
        )
        return out

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states
        q, kv, qr = self._project_q_kv(hidden_states)
        pool = ctx.token_to_kv_pool
        self._insert_swa_cache(
            q=q,
            kv=kv,
            swa_kv_cache=pool.get_swa_kv_buffer(self.layer_index),
            slot_mapping=out_cache_loc,
            positions=positions,
            block_size=pool.swa_block_size,
        )
        if self.compressor is not None:
            self.compressor(
                hidden_states=hidden_states,
                positions=positions,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                layer_index=self.layer_index,
                cos_sin_cache=self._cos_sin_cache(),
            )
        topk_indices = None
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states=hidden_states,
                qr=qr,
                positions=positions,
                ctx=ctx,
                out_cache_loc=out_cache_loc,
                layer_index=self.layer_index,
                cos_sin_cache=self._cos_sin_cache(),
            )
        backend_decode = getattr(
            ctx.attn_backend,
            "forward_deepseek_v4_decode",
            None,
        )
        backend_prefill = getattr(
            ctx.attn_backend,
            "forward_deepseek_v4_prefill",
            None,
        )
        if (
            backend_decode is not None
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_decode()
        ):
            attn_output = backend_decode(
                q=q,
                positions=positions,
                token_to_kv_pool=pool,
                layer_id=self.layer_index,
                kind=self.attention_kind,
                compress_ratio=self.compress_ratio,
                num_local_heads=self.num_local_heads,
                padded_heads=self.padded_heads,
                head_dim=self.head_dim,
                window_size=self.layout.swa_window,
                softmax_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_indices=topk_indices,
            )
        elif (
            backend_prefill is not None
            and ctx.forward_mode is not None
            and ctx.forward_mode.is_extend()
        ):
            attn_output = backend_prefill(
                q=q,
                positions=positions,
                token_to_kv_pool=pool,
                layer_id=self.layer_index,
                compress_ratio=self.compress_ratio,
                num_local_heads=self.num_local_heads,
                padded_heads=self.padded_heads,
                head_dim=self.head_dim,
                window_size=self.layout.swa_window,
                softmax_scale=self.scale,
                attn_sink=self.attn_sink,
                topk_indices=topk_indices,
            )
        else:
            attn_output = self._forward_flashmla_sparse(q, positions, ctx, topk_indices)
        return self._project_attention_output(attn_output, positions)


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.mapping = mapping
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + self.hc_mult) * self.hc_mult
        hc_dim = self.hc_mult * config.hidden_size
        self.attn = DeepseekV4Attention(
            config, mapping, layer_id, quant_config, add_prefix("attn", prefix)
        )
        self.ffn = DeepseekV4MoE(
            config, mapping, quant_config, layer_id, add_prefix("ffn", prefix)
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=True,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )

    def _hc_pre(
        self, x: torch.Tensor, fn: torch.Tensor, scale: torch.Tensor, base: torch.Tensor
    ):
        return mhc_pre(
            x,
            fn,
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )

    def _pre_mlp_input_ids_comm(
        self, input_ids: torch.Tensor, ctx: ForwardContext
    ) -> torch.Tensor:
        if not self.mapping.moe.has_tp_ep:
            return input_ids
        if self.comm_manager.use_all_reduce(is_moe=True):
            return input_ids

        token_counts = self.comm_manager.moe_tp_ep_group_scattered_num_tokens(ctx)
        max_tokens = max(token_counts)
        padded = torch.empty(
            (max_tokens,), device=input_ids.device, dtype=input_ids.dtype
        )
        padded[: input_ids.shape[0]].copy_(input_ids)
        if input_ids.shape[0] < max_tokens:
            padded[input_ids.shape[0] :].zero_()

        gathered = [torch.empty_like(padded) for _ in token_counts]
        group = pg_manager.get_process_group("nccl", self.mapping.moe.tp_ep_group)
        torch.distributed.all_gather(gathered, padded, group=group)
        return torch.cat(
            [tokens[:count] for tokens, count in zip(gathered, token_counts)], dim=0
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )
        hidden_states = self.attn_norm(hidden_states)
        hidden_states = self.attn(positions, hidden_states, ctx, out_cache_loc)
        hidden_states = mhc_post(hidden_states, residual, post, comb)

        residual = hidden_states
        hidden_states, post, comb = self._hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )
        hidden_states = self.ffn_norm(hidden_states)
        ffn_input_ids = input_ids
        hidden_states = self.comm_manager.pre_mlp_comm(hidden_states, ctx)
        if self.ffn.gate.is_hash_moe:
            ffn_input_ids = self._pre_mlp_input_ids_comm(input_ids, ctx)
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )
        hidden_states = self.ffn(
            hidden_states,
            ffn_input_ids,
            num_global_tokens,
            max_num_tokens_per_gpu,
        )
        hidden_states, _ = self.comm_manager.post_mlp_comm(hidden_states, None, ctx)
        hidden_states = mhc_post(hidden_states, residual, post, comb)
        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mapping = mapping
        self.hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        self.rms_norm_eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV4DecoderLayer(
                    config,
                    layer_id,
                    mapping,
                    quant_config,
                    add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hc_dim = config.hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(config.hc_mult, hc_dim, dtype=torch.float32),
            requires_grad=False,
        )
        self.hc_head_base = nn.Parameter(
            torch.empty(config.hc_mult, dtype=torch.float32), requires_grad=False
        )
        self.hc_head_scale = nn.Parameter(
            torch.empty(1, dtype=torch.float32), requires_grad=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        out_cache_loc: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = input_embeds
        if hidden_states is None:
            hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        for layer in self.layers:
            hidden_states = layer(
                positions, hidden_states, ctx, out_cache_loc, input_ids
            )
        hidden_states = hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
            self.rms_norm_eps,
            self.hc_eps,
        )
        hidden_states = self.norm(hidden_states)
        return hidden_states, None


class DeepseekV4ForCausalLM(BaseCausalLM):
    model_cls = DeepseekV4Model

    def get_stacked_params_mapping(self):
        return [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]

    @staticmethod
    def _map_weight_name(name: str) -> str:
        if name.startswith("layers."):
            name = "model." + name
        elif name.startswith("embed."):
            name = name.replace("embed.", "model.embed_tokens.", 1)
        elif name.startswith("norm."):
            name = "model." + name
        elif name.startswith("hc_head"):
            name = "model." + name
        elif name == "head.weight":
            name = "lm_head.weight"
        if ".shared_experts.w2" in name:
            name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
        if ".ffn.gate.bias" in name:
            name = name.replace(".ffn.gate.bias", ".ffn.gate.e_score_correction_bias")
        if re.search(r"\.experts\.\d+\.w[123]\.scale$", name):
            name = name.replace(".scale", ".weight_scale")
        elif name.endswith(".scale"):
            name = name[:-6] + ".weight_scale_inv"
        return name

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = self.get_stacked_params_mapping()
        params_dict = dict(self.named_parameters())
        moe_loader = build_moe_checkpoint_loader(
            params_dict=params_dict,
            expert_schema=ExpertCheckpointSchema(
                gate_proj_name="w1",
                down_proj_name="w2",
                up_proj_name="w3",
            ),
            num_experts=self.config.n_routed_experts,
            ep_rank=self.mapping.moe.ep_rank,
            ep_size=self.mapping.moe.ep_size,
        )
        for raw_name, loaded_weight in weights:
            name = self._map_weight_name(raw_name)
            if name.startswith("mtp."):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or ".experts." in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is None:
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if moe_loader.matches(name):
                    moe_loader.load(name, loaded_weight)
                    continue
                param = params_dict.get(name)
                if param is None:
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
        self.post_load_weights()

    def post_load_weights(self):
        for module in self.modules():
            if isinstance(module, DeepseekV4Compressor):
                module.process_weights_after_loading()
            elif isinstance(module, MoELayer):
                module.process_weights_after_loading(module)

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation

        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=getattr(config, "n_group", 0),
        )


EntryClass = [
    DeepseekV4ForCausalLM,
]
