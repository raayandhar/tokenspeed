# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#
# DeepSeek V4 compressor/indexer/attention helpers keep the boundary unfused
# and framework-local so the correctness contract can be tested before the
# optimized fused kernel lands.

"""DeepSeek V4 attention kernel boundaries.

Keep the model layer independent from the CUDA extension import details. The
runtime requires TokenSpeed's own built DeepSeek V4 attention op.
"""

from __future__ import annotations

import math

import torch

QNORM_ROPE_KV_INSERT_OP = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_ROPE_DIM = 64
DEEPSEEK_V4_NOPE_DIM = DEEPSEEK_V4_HEAD_DIM - DEEPSEEK_V4_ROPE_DIM
DEEPSEEK_V4_FP8_MAX = 448.0
DEEPSEEK_V4_FP8_QUANT_BLOCK = 64
DEEPSEEK_V4_MXFP4_BLOCK_SIZE = 32
DEEPSEEK_V4_INDEXER_DIM = 128
DEEPSEEK_V4_SWA_TOKEN_STRIDE = DEEPSEEK_V4_NOPE_DIM + DEEPSEEK_V4_ROPE_DIM * 2
DEEPSEEK_V4_SWA_SCALE_DIM = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK + 1
DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES = DEEPSEEK_V4_INDEXER_DIM // 2
DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM = (
    DEEPSEEK_V4_INDEXER_DIM // DEEPSEEK_V4_MXFP4_BLOCK_SIZE
)


class DeepseekV4AttentionOpUnavailable(RuntimeError):
    pass


def _get_tokenspeed_op() -> object | None:
    try:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            fused_qnorm_rope_kv_insert as op,
        )
    except Exception:
        return None
    return op


def _has_tokenspeed_op() -> bool:
    try:
        from tokenspeed_kernel.thirdparty.cuda.deepseek_v4_attention import (
            has_fused_qnorm_rope_kv_insert as has_op,
        )
    except Exception:
        return False
    return has_op()


def has_fused_qnorm_rope_kv_insert() -> bool:
    return _has_tokenspeed_op()


def _require_op():
    tokenspeed_op = _get_tokenspeed_op()
    if tokenspeed_op is not None and _has_tokenspeed_op():
        return tokenspeed_op
    raise DeepseekV4AttentionOpUnavailable(
        f"DeepSeek V4 fused SWA cache insert op {QNORM_ROPE_KV_INSERT_OP} "
        "is unavailable. Build `tokenspeed-kernel/python` so the "
        "deepseek_v4_attention CUDA library is present before running this path."
    )


def fused_qnorm_rope_kv_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_eps: float,
    block_size: int,
) -> None:
    """Run the DeepSeek V4 fused SWA cache insert op.

    Expected contract:
    - q: [tokens, local_heads, 512], mutated in place by RMSNorm/RoPE
    - kv: [tokens, 512], source KV latent before RoPE/quant insert
    - swa_kv_cache_2d: uint8 cache blocks flattened as [num_blocks, block_bytes]
    - slot_mapping: output token slots in the paged SWA cache
    - positions: absolute token positions
    """

    op = _require_op()
    op(
        q,
        kv,
        swa_kv_cache_2d,
        slot_mapping,
        positions.to(torch.int64),
        cos_sin_cache,
        rms_norm_eps,
        block_size,
    )


def _apply_gptj_rope_tail(
    x: torch.Tensor,
    position: int,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos_sin = cos_sin_cache[position]
    cos = cos_sin[:half_rope].float()
    sin = cos_sin[half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    out[..., nope_dim::2] = even * cos - odd * sin
    out[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    return out


def _apply_inverse_gptj_rope_tail(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    half_rope = rope_dim // 2
    nope_dim = x.shape[-1] - rope_dim
    cos = cos_sin_cache[positions.long(), :half_rope].float()
    sin = cos_sin_cache[positions.long(), half_rope:rope_dim].float()
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos + odd * sin
    out[..., nope_dim + 1 :: 2] = odd * cos - even * sin
    return out


def _encode_ue8m0_exponent(exponent: float) -> int:
    return int(max(min(exponent + 127, 255), 0))


def _fp8_e4m3_ue8m0_bytes(block: torch.Tensor) -> tuple[torch.Tensor, int]:
    absmax = max(float(block.detach().abs().max()), 1.0e-4)
    exponent = math.ceil(math.log2(absmax / DEEPSEEK_V4_FP8_MAX))
    scaled = torch.clamp(
        block * (2.0**-exponent),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), _encode_ue8m0_exponent(
        exponent
    )


def _fp8_e4m3_pow2_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(block.detach().abs().max()) / DEEPSEEK_V4_FP8_MAX, 1.0e-10)
    scale = 2.0 ** math.ceil(math.log2(scale))
    scaled = torch.clamp(block / scale, -DEEPSEEK_V4_FP8_MAX, DEEPSEEK_V4_FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), block.new_tensor(scale)


def _fp8_e4m3_pow2_dequant_rows(
    rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows_f = rows.float()
    scale = (rows_f.detach().abs().amax(dim=-1) / DEEPSEEK_V4_FP8_MAX).clamp_min(
        1.0e-10
    )
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    scaled = torch.clamp(
        rows_f / scale.unsqueeze(-1),
        -DEEPSEEK_V4_FP8_MAX,
        DEEPSEEK_V4_FP8_MAX,
    )
    dequant = scaled.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return dequant, scale


def _e2m1_nibbles(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.clamp(x.abs(), max=6.0)
    code = torch.zeros_like(abs_x, dtype=torch.uint8)
    for idx, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        code = torch.where(abs_x > boundary, idx + 1, code)
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    magnitude = table[(nibbles & 0x7).long()]
    sign = torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    return magnitude * sign


def _mxfp4_e2m1_ue8m0_bytes(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if block.numel() != DEEPSEEK_V4_MXFP4_BLOCK_SIZE:
        raise ValueError(
            f"MXFP4 block must have {DEEPSEEK_V4_MXFP4_BLOCK_SIZE} values, "
            f"got {block.numel()}"
        )
    even = block.float()[0::2]
    odd = block.float()[1::2]
    absmax = max(
        float(torch.maximum(even.detach().abs().max(), odd.detach().abs().max())),
        1.0e-4,
    )
    exponent = min(max(math.ceil(math.log2(absmax / 6.0)), -127), 127)
    inv_scale = 2.0**-exponent
    lo = _e2m1_nibbles(even * inv_scale)
    hi = _e2m1_nibbles(odd * inv_scale)
    return lo | (hi << 4), block.new_tensor(exponent + 127, dtype=torch.uint8)


def _mxfp4_e2m1_ue8m0_dequant_rows(rows: torch.Tensor) -> torch.Tensor:
    orig_shape = rows.shape
    if orig_shape[-1] % DEEPSEEK_V4_MXFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"MXFP4 rows require last dim divisible by {DEEPSEEK_V4_MXFP4_BLOCK_SIZE}, "
            f"got {orig_shape[-1]}"
        )
    blocks = rows.float().reshape(
        -1, orig_shape[-1] // DEEPSEEK_V4_MXFP4_BLOCK_SIZE, DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    )
    absmax = blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / 6.0)).clamp(-127, 127)
    scale = torch.pow(2.0, exponent)
    nibbles = _e2m1_nibbles(blocks / scale.unsqueeze(-1))
    dequant = _e2m1_values(nibbles) * scale.unsqueeze(-1)
    return dequant.reshape(orig_shape)


def _deepseek_v4_hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    try:
        from tokenspeed_kernel.thirdparty.fast_hadamard_transform import (
            hadamard_transform,
        )
    except Exception as exc:
        raise DeepseekV4AttentionOpUnavailable(
            "DeepSeek V4 CSA indexer requires fast_hadamard_transform. "
            "Build/install `tokenspeed-kernel/python` before serving V4."
        ) from exc

    shape = x.shape
    rotated = hadamard_transform(
        x.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    )
    return rotated.reshape(shape)


def _fp8_e4m3_ue8m0_dequant(block: torch.Tensor) -> torch.Tensor:
    fp8_bytes, encoded_scale = _fp8_e4m3_ue8m0_bytes(block.float())
    scale = 2.0 ** (encoded_scale - 127)
    return fp8_bytes.view(torch.float8_e4m3fn).float() * scale


def deepseek_v4_inv_rope_fp8_dequant_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = DEEPSEEK_V4_NOPE_DIM,
    rope_dim: int = DEEPSEEK_V4_ROPE_DIM,
    quant_group_size: int = 128,
) -> torch.Tensor:
    """Inverse-RoPE and FP8 block-quantize/dequantize V4 attention output.

    This reference helper keeps the FP8 rounding/scaling contract but returns
    BF16 dequantized values so TokenSpeed can use an auditable PyTorch grouped
    matmul until the fused `fp8_einsum` boundary is added.
    """

    if o.dim() != 3:
        raise ValueError(f"o must be [tokens, heads, dim], got {tuple(o.shape)}")
    if o.shape[1] != n_groups * heads_per_group:
        raise ValueError(
            f"heads={o.shape[1]} does not match n_groups={n_groups} "
            f"* heads_per_group={heads_per_group}"
        )
    if o.shape[2] != nope_dim + rope_dim:
        raise ValueError(f"head dim must be {nope_dim + rope_dim}, got {o.shape[2]}")
    if (heads_per_group * o.shape[2]) % quant_group_size != 0:
        raise ValueError("grouped output width must be divisible by quant_group_size")

    inv = _apply_inverse_gptj_rope_tail(o, positions, cos_sin_cache, rope_dim)
    grouped = inv.reshape(o.shape[0], n_groups, heads_per_group * o.shape[2])
    out = torch.empty_like(grouped, dtype=torch.float32)
    for token_idx in range(grouped.shape[0]):
        for group_idx in range(grouped.shape[1]):
            row = grouped[token_idx, group_idx]
            for start in range(0, row.numel(), quant_group_size):
                out[token_idx, group_idx, start : start + quant_group_size] = (
                    _fp8_e4m3_ue8m0_dequant(row[start : start + quant_group_size])
                )
    return out.to(o.dtype)


def deepseek_v4_inv_rope_reference(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = DEEPSEEK_V4_NOPE_DIM,
    rope_dim: int = DEEPSEEK_V4_ROPE_DIM,
) -> torch.Tensor:
    """Inverse-RoPE and group V4 attention output without FP8 activation rounding."""

    if o.dim() != 3:
        raise ValueError(f"o must be [tokens, heads, dim], got {tuple(o.shape)}")
    if o.shape[1] != n_groups * heads_per_group:
        raise ValueError(
            f"heads={o.shape[1]} does not match n_groups={n_groups} "
            f"* heads_per_group={heads_per_group}"
        )
    if o.shape[2] != nope_dim + rope_dim:
        raise ValueError(f"head dim must be {nope_dim + rope_dim}, got {o.shape[2]}")

    inv = _apply_inverse_gptj_rope_tail(o, positions, cos_sin_cache, rope_dim)
    grouped = inv.reshape(o.shape[0], n_groups, heads_per_group * o.shape[2])
    return grouped.to(o.dtype)


def dequantize_deepseek_v4_fp8_ds_mla_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize DeepSeek V4 `fp8_ds_mla` rows selected by global slots."""

    min_stride = block_size * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), DEEPSEEK_V4_HEAD_DIM)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.bfloat16)

    flat_cache = cache_2d.reshape(-1)
    num_nope_blocks = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK

    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_SWA_TOKEN_STRIDE
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        + pos * DEEPSEEK_V4_SWA_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_SWA_TOKEN_STRIDE, device=cache_2d.device, dtype=torch.int64
        )[None, :]
    )
    row_bytes = flat_cache[value_offsets]
    nope = row_bytes[:, :DEEPSEEK_V4_NOPE_DIM].contiguous().view(torch.float8_e4m3fn)

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(num_nope_blocks, device=cache_2d.device, dtype=torch.int64)[
            None, :
        ]
    )
    scales = torch.pow(2.0, flat_cache[scale_offsets].to(torch.int32) - 127)
    scales = scales.float().repeat_interleave(DEEPSEEK_V4_FP8_QUANT_BLOCK, dim=1)

    rope = row_bytes[:, DEEPSEEK_V4_NOPE_DIM:DEEPSEEK_V4_SWA_TOKEN_STRIDE].contiguous()
    out = torch.cat([nope.float() * scales, rope.view(torch.bfloat16).float()], dim=1)
    out = out.to(torch.bfloat16)
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def deepseek_v4_prepare_indexer_q_reference(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply indexer Q RoPE and quant/dequant with folded weight rules."""

    if index_q.dim() != 3 or index_q.shape[-1] != DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_q must be [tokens, heads, {DEEPSEEK_V4_INDEXER_DIM}], "
            f"got {tuple(index_q.shape)}"
        )
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != index_q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {tuple(weights.shape)}")

    rotated = index_q.float().clone()
    half_rope = DEEPSEEK_V4_ROPE_DIM // 2
    nope_dim = index_q.shape[-1] - DEEPSEEK_V4_ROPE_DIM
    cos = cos_sin_cache[positions.long(), :half_rope].float().unsqueeze(1)
    sin = (
        cos_sin_cache[positions.long(), half_rope:DEEPSEEK_V4_ROPE_DIM]
        .float()
        .unsqueeze(1)
    )
    even = rotated[..., nope_dim::2].clone()
    odd = rotated[..., nope_dim + 1 :: 2].clone()
    rotated[..., nope_dim::2] = even * cos - odd * sin
    rotated[..., nope_dim + 1 :: 2] = even * sin + odd * cos

    rotated = _deepseek_v4_hadamard_rotate(rotated).float()
    weights_out = weights.float().clone()
    if use_fp4:
        q_out = _mxfp4_e2m1_ue8m0_dequant_rows(rotated)
        weights_out *= softmax_scale * head_scale
    else:
        q_out, q_scale = _fp8_e4m3_pow2_dequant_rows(rotated)
        weights_out *= q_scale * softmax_scale * head_scale
    return q_out.to(index_q.dtype), weights_out


def _write_fp8_ds_mla_cache_row(
    normed: torch.Tensor,
    position: int,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot: int,
    kv_cache_block_size: int,
    compress_ratio: int,
) -> None:
    quant_input = normed.to(torch.bfloat16).float()
    kv_block_idx = kv_slot // kv_cache_block_size
    kv_pos_in_block = kv_slot % kv_cache_block_size
    token_base = (
        kv_block_idx * kv_cache_2d.stride(0)
        + kv_pos_in_block * DEEPSEEK_V4_SWA_TOKEN_STRIDE
    )
    scale_base = (
        kv_block_idx * kv_cache_2d.stride(0)
        + kv_cache_block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        + kv_pos_in_block * DEEPSEEK_V4_SWA_SCALE_DIM
    )
    flat_cache = kv_cache_2d.reshape(-1)
    num_nope_blocks = DEEPSEEK_V4_NOPE_DIM // DEEPSEEK_V4_FP8_QUANT_BLOCK
    for block_id in range(num_nope_blocks):
        lo = block_id * DEEPSEEK_V4_FP8_QUANT_BLOCK
        hi = lo + DEEPSEEK_V4_FP8_QUANT_BLOCK
        fp8_bytes, encoded_scale = _fp8_e4m3_ue8m0_bytes(quant_input[lo:hi])
        flat_cache[token_base + lo : token_base + hi].copy_(fp8_bytes)
        flat_cache[scale_base + block_id] = encoded_scale
    flat_cache[scale_base + num_nope_blocks] = 0

    compressed_position = (position // compress_ratio) * compress_ratio
    rotated = _apply_gptj_rope_tail(
        normed, compressed_position, cos_sin_cache, DEEPSEEK_V4_ROPE_DIM
    ).to(torch.bfloat16)
    flat_cache[
        token_base + DEEPSEEK_V4_NOPE_DIM : token_base + DEEPSEEK_V4_SWA_TOKEN_STRIDE
    ].copy_(rotated[DEEPSEEK_V4_NOPE_DIM:].view(torch.uint8))


def save_deepseek_v4_compressor_state(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    block_size: int,
    compress_ratio: int,
) -> None:
    """Save DeepSeek V4 compressor residual state into paged SWA-style cache.

    This correctness-first state write packs `[kv_state, score_state]`, each
    with width `coff * head_dim`; score state includes the APE row selected by
    `position % compress_ratio`.
    """

    if kv.shape != score.shape:
        raise ValueError(
            f"kv and score shapes must match, got {kv.shape} vs {score.shape}"
        )
    if kv.dim() != 2:
        raise ValueError(f"kv/score must be [tokens, state_width], got {kv.shape}")
    if state_cache.dim() != 3:
        raise ValueError(
            "state_cache must be [blocks, block_size, 2 * state_width], "
            f"got {state_cache.shape}"
        )
    if block_size != state_cache.shape[1]:
        raise ValueError(
            f"block_size={block_size} does not match "
            f"state_cache.shape[1]={state_cache.shape[1]}"
        )
    state_width = kv.shape[-1]
    if state_cache.shape[-1] != state_width * 2:
        raise ValueError(
            f"state_cache last dim must be {state_width * 2}, "
            f"got {state_cache.shape[-1]}"
        )
    if ape.shape != (compress_ratio, state_width):
        raise ValueError(
            f"ape must be [{compress_ratio}, {state_width}], got {tuple(ape.shape)}"
        )

    num_actual = min(slot_mapping.numel(), kv.shape[0])
    if num_actual == 0:
        return
    if num_actual <= 2:
        for token_idx in range(num_actual):
            slot = int(slot_mapping[token_idx].item())
            if slot < 0:
                continue
            block_idx = slot // block_size
            pos_in_block = slot % block_size
            position = int(positions[token_idx].item())
            row = state_cache[block_idx, pos_in_block]
            row[:state_width].copy_(kv[token_idx].float())
            score_row = score[token_idx].float()
            if (
                compress_ratio == 4
                and state_width == ape.shape[1]
                and state_width % 2 == 0
            ):
                head_dim = state_width // 2
                ape_slots = ape.view(-1, head_dim)
                slot_idx = position % compress_ratio
                scored = score_row.clone()
                scored[:head_dim] += ape_slots[slot_idx].float()
                scored[head_dim:] += ape_slots[slot_idx + compress_ratio].float()
                row[state_width:].copy_(scored)
            else:
                row[state_width:].copy_(
                    score_row + ape[position % compress_ratio].float()
                )
        return

    slots = slot_mapping[:num_actual].to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any()):
        return

    valid_slots = slots[valid]
    block_idx = torch.div(valid_slots, block_size, rounding_mode="floor")
    pos_in_block = valid_slots % block_size
    state_cache[block_idx, pos_in_block, :state_width] = kv[:num_actual][valid].float()

    score_rows = score[:num_actual][valid].float()
    valid_positions = positions[:num_actual][valid].to(torch.int64)
    if compress_ratio == 4 and state_width == ape.shape[1] and state_width % 2 == 0:
        head_dim = state_width // 2
        ape_slots = ape.view(-1, head_dim).float()
        slot_idx = valid_positions % compress_ratio
        scored = score_rows.clone()
        scored[:, :head_dim] += ape_slots[slot_idx]
        scored[:, head_dim:] += ape_slots[slot_idx + compress_ratio]
    else:
        scored = score_rows + ape[valid_positions % compress_ratio].float()
    state_cache[block_idx, pos_in_block, state_width:] = scored


def write_deepseek_v4_indexer_fp8_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write FP8 indexer keys using `[values | fp32 scale]` page layout."""

    if index_k.dim() != 2 or index_k.shape[-1] != 128:
        raise ValueError(f"index_k must be [tokens, 128], got {tuple(index_k.shape)}")
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < block_size * 132:
        raise ValueError(
            f"cache_2d must be [pages, >= {block_size * 132}], "
            f"got {tuple(cache_2d.shape)}"
        )

    flat_cache = cache_2d.reshape(-1)
    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    for token_idx in range(num_actual):
        slot = int(slot_mapping[token_idx].item())
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        page_base = page * cache_2d.stride(0)
        value_base = page_base + pos * 128
        scale_base = page_base + block_size * 128 + pos * 4
        q_bytes, scale = _fp8_e4m3_pow2_bytes(index_k[token_idx].float())
        flat_cache[value_base : value_base + 128].copy_(q_bytes)
        flat_cache[scale_base : scale_base + 4].copy_(
            scale.reshape(1).view(torch.uint8)
        )


def write_deepseek_v4_indexer_mxfp4_cache(
    index_k: torch.Tensor,
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> None:
    """Write MXFP4 indexer keys using the `[values | ue8m0 scales]` layout."""

    if index_k.dim() != 2 or index_k.shape[-1] != DEEPSEEK_V4_INDEXER_DIM:
        raise ValueError(
            f"index_k must be [tokens, {DEEPSEEK_V4_INDEXER_DIM}], "
            f"got {tuple(index_k.shape)}"
        )
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    min_stride = block_size * (
        DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    num_actual = min(slot_mapping.numel(), index_k.shape[0])
    if num_actual == 0:
        return

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping[:num_actual].to(torch.int64)
    valid = slots >= 0
    if not bool(valid.any()):
        return

    valid_slots = slots[valid]
    valid_rows = index_k[:num_actual][valid].float()
    blocks = valid_rows.reshape(
        -1, DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM, DEEPSEEK_V4_MXFP4_BLOCK_SIZE
    )
    absmax = blocks.detach().abs().amax(dim=-1).clamp_min(1.0e-4)
    exponent = torch.ceil(torch.log2(absmax / 6.0)).clamp(-127, 127)
    inv_scale = torch.pow(2.0, -exponent)
    nibbles = _e2m1_nibbles(blocks * inv_scale.unsqueeze(-1))
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(
        -1, DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    )
    scale_bytes = (exponent.to(torch.int32) + 127).to(torch.uint8)

    pages = torch.div(valid_slots, block_size, rounding_mode="floor")
    pos = valid_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        + pos * DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    flat_cache[value_offsets] = packed
    flat_cache[scale_offsets] = scale_bytes


def read_deepseek_v4_indexer_mxfp4_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize MXFP4 indexer cache rows selected by `slot_mapping`."""

    min_stride = block_size * (
        DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES + DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out_shape = (slot_mapping.numel(), DEEPSEEK_V4_INDEXER_DIM)
    if slot_mapping.numel() == 0:
        return torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)

    flat_cache = cache_2d.reshape(-1)
    slots = slot_mapping.to(torch.int64)
    valid = slots >= 0
    safe_slots = torch.where(valid, slots, torch.zeros_like(slots))
    pages = torch.div(safe_slots, block_size, rounding_mode="floor")
    pos = safe_slots % block_size
    page_base = pages * cache_2d.stride(0)
    value_base = page_base + pos * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
    scale_base = (
        page_base
        + block_size * DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES
        + pos * DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM
    )

    value_offsets = (
        value_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_VALUE_BYTES,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    packed = flat_cache[value_offsets]

    scale_offsets = (
        scale_base[:, None]
        + torch.arange(
            DEEPSEEK_V4_INDEXER_MXFP4_SCALE_DIM,
            device=cache_2d.device,
            dtype=torch.int64,
        )[None, :]
    )
    scales = torch.pow(2.0, flat_cache[scale_offsets].to(torch.int32) - 127)
    byte_scales = scales.float().repeat_interleave(
        DEEPSEEK_V4_MXFP4_BLOCK_SIZE // 2, dim=1
    )

    even = _e2m1_values(packed & 0xF) * byte_scales
    odd = _e2m1_values(packed >> 4) * byte_scales
    out = torch.empty(out_shape, device=cache_2d.device, dtype=torch.float32)
    out[:, 0::2] = even
    out[:, 1::2] = odd
    return torch.where(valid[:, None], out, torch.zeros_like(out))


def read_deepseek_v4_indexer_fp8_cache(
    cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Dequantize FP8 indexer cache rows selected by `slot_mapping`."""

    min_stride = block_size * (DEEPSEEK_V4_INDEXER_DIM + 4)
    if cache_2d.dtype != torch.uint8:
        raise TypeError(f"cache_2d must be uint8, got {cache_2d.dtype}")
    if cache_2d.dim() != 2 or cache_2d.shape[1] < min_stride:
        raise ValueError(
            f"cache_2d must be [pages, >= {min_stride}], got {tuple(cache_2d.shape)}"
        )

    out = torch.zeros(
        slot_mapping.numel(),
        DEEPSEEK_V4_INDEXER_DIM,
        device=cache_2d.device,
        dtype=torch.float32,
    )
    flat_cache = cache_2d.reshape(-1)
    for token_idx, raw_slot in enumerate(slot_mapping.tolist()):
        slot = int(raw_slot)
        if slot < 0:
            continue
        page = slot // block_size
        pos = slot % block_size
        page_base = page * cache_2d.stride(0)
        value_base = page_base + pos * DEEPSEEK_V4_INDEXER_DIM
        scale_base = page_base + block_size * DEEPSEEK_V4_INDEXER_DIM + pos * 4
        scale = flat_cache[scale_base : scale_base + 4].view(torch.float32)[0]
        values = flat_cache[value_base : value_base + DEEPSEEK_V4_INDEXER_DIM].view(
            torch.float8_e4m3fn
        )
        out[token_idx].copy_(values.float() * scale)
    return out


def deepseek_v4_indexer_topk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference weighted ReLU MQA top-k for the CSA sparse indexer."""

    if q.dim() != 3 or k.dim() != 2:
        raise ValueError(
            f"expected q [tokens, heads, dim], k [kv, dim], got {q.shape}, {k.shape}"
        )
    if q.shape[-1] != k.shape[-1]:
        raise ValueError(f"q/k dims must match, got {q.shape[-1]} and {k.shape[-1]}")
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    if weights.shape != q.shape[:2]:
        raise ValueError(f"weights must be [tokens, heads], got {weights.shape}")

    logits = torch.einsum("thd,kd->thk", q.float(), k.float()).relu()
    logits = (logits * weights.float().unsqueeze(-1)).sum(dim=1)
    if lengths is not None:
        if row_starts is None:
            row_starts = torch.zeros_like(lengths)
        cols = torch.arange(k.shape[0], device=k.device)
        valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
            cols.unsqueeze(0) < (row_starts + lengths).unsqueeze(1)
        )
        logits = logits.masked_fill(~valid, -float("inf"))
    return torch.topk(logits, k=top_k, dim=-1, sorted=False).indices.to(torch.int32)


def _compress_v4_state_window(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    token_idx: int,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
) -> torch.Tensor | None:
    state_slot = int(compressor_slot_mapping[token_idx].item())
    if state_slot < 0:
        return None
    position = int(positions[token_idx].item())
    if (position + 1) % compress_ratio != 0:
        return None

    state_width = state_cache.shape[-1] // 2
    window = (2 if overlap else 1) * compress_ratio
    req_idx = int(token_to_req_indices[token_idx].item())
    start = position - window + 1
    kv_rows = []
    score_rows = []
    for offset in range(window):
        pos = start + offset
        if pos < 0:
            continue
        table_idx = pos // compressor_block_size
        if table_idx >= block_table.shape[1]:
            continue
        block_number = int(block_table[req_idx, table_idx].item())
        if block_number < 0:
            continue
        head_offset = head_dim if overlap and offset >= compress_ratio else 0
        row = state_cache[block_number, pos % compressor_block_size]
        kv_rows.append(row[head_offset : head_offset + head_dim].float())
        score_rows.append(
            row[
                state_width + head_offset : state_width + head_offset + head_dim
            ].float()
        )
    if not kv_rows:
        return None

    kv_stack = torch.stack(kv_rows, dim=0)
    score_stack = torch.stack(score_rows, dim=0)
    weights = torch.softmax(score_stack, dim=0)
    compressed = torch.sum(kv_stack * weights, dim=0)
    variance = compressed.square().sum() / float(head_dim)
    normed = compressed * torch.rsqrt(variance + rms_norm_eps)
    return normed * rms_norm_weight.float()


def deepseek_v4_hca_compress_kv_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int = 128,
) -> None:
    """Compress HCA state, normalize/RoPE/FP8-quantize, and insert KV cache.

    The HCA path writes one compressed cache entry only at positions where
    `(position + 1) % 128 == 0`. This path is deliberately unfused so
    TokenSpeed has an auditable correctness boundary before the optimized
    kernel lands.
    """

    if compress_ratio != 128:
        raise ValueError(
            f"HCA cache insert requires compress_ratio=128, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    if state_width != DEEPSEEK_V4_HEAD_DIM:
        raise ValueError(
            f"HCA state width must be {DEEPSEEK_V4_HEAD_DIM}, got {state_width}"
        )
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    min_block_stride = kv_cache_block_size * (
        DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    for token_idx in range(num_actual):
        state_slot = int(compressor_slot_mapping[token_idx].item())
        if state_slot < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue

        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=False,
        )
        if normed is None:
            continue

        _write_fp8_ds_mla_cache_row(
            normed,
            position,
            cos_sin_cache,
            kv_cache_2d,
            kv_slot,
            kv_cache_block_size,
            compress_ratio,
        )


def deepseek_v4_csa_compress_kv_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    compress_ratio: int = 4,
) -> None:
    """Compress CSA state and insert one `fp8_ds_mla` row per 4 tokens.

    CSA uses overlap: the compression window spans eight token positions and
    selects the first 512-wide slice from the older four positions and the
    second slice from the newer four positions before the softmax-weighted sum.
    """

    if compress_ratio != 4:
        raise ValueError(
            f"CSA cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    expected_width = DEEPSEEK_V4_HEAD_DIM * 2
    if state_width != expected_width:
        raise ValueError(f"CSA state width must be {expected_width}, got {state_width}")
    if compressor_block_size != state_cache.shape[1]:
        raise ValueError(
            "compressor_block_size must match state_cache page size, "
            f"got {compressor_block_size} vs {state_cache.shape[1]}"
        )
    min_block_stride = kv_cache_block_size * (
        DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM
    )
    if kv_cache_2d.dim() != 2 or kv_cache_2d.shape[1] < min_block_stride:
        raise ValueError(
            f"kv_cache_2d must be [blocks, >= {min_block_stride}] uint8, "
            f"got {tuple(kv_cache_2d.shape)}"
        )
    if kv_cache_2d.dtype != torch.uint8:
        raise TypeError(f"kv_cache_2d must be uint8, got {kv_cache_2d.dtype}")

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    for token_idx in range(num_actual):
        state_slot = int(compressor_slot_mapping[token_idx].item())
        if state_slot < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue

        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_HEAD_DIM,
            overlap=True,
        )
        if normed is None:
            continue
        _write_fp8_ds_mla_cache_row(
            normed,
            position,
            cos_sin_cache,
            kv_cache_2d,
            kv_slot,
            kv_cache_block_size,
            compress_ratio,
        )


def deepseek_v4_csa_indexer_cache_insert(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    compressor_slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    compressor_block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache_2d: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    use_fp4_cache: bool,
    compress_ratio: int = 4,
) -> None:
    """Compress CSA indexer state and insert FP8/MXFP4 indexer cache rows."""

    if compress_ratio != 4:
        raise ValueError(
            f"CSA indexer cache insert requires compress_ratio=4, got {compress_ratio}"
        )
    if state_cache.dim() != 3:
        raise ValueError(f"state_cache must be 3D, got {tuple(state_cache.shape)}")
    state_width = state_cache.shape[-1] // 2
    expected_width = DEEPSEEK_V4_INDEXER_DIM * 2
    if state_width != expected_width:
        raise ValueError(
            f"CSA indexer state width must be {expected_width}, got {state_width}"
        )

    num_actual = min(compressor_slot_mapping.numel(), positions.numel())
    for token_idx in range(num_actual):
        kv_slot = int(kv_slot_mapping[token_idx].item())
        if kv_slot < 0:
            continue
        position = int(positions[token_idx].item())
        normed = _compress_v4_state_window(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=compressor_slot_mapping,
            block_table=block_table,
            compressor_block_size=compressor_block_size,
            rms_norm_weight=rms_norm_weight,
            rms_norm_eps=rms_norm_eps,
            token_idx=token_idx,
            compress_ratio=compress_ratio,
            head_dim=DEEPSEEK_V4_INDEXER_DIM,
            overlap=True,
        )
        if normed is None:
            continue
        compressed_position = (position // compress_ratio) * compress_ratio
        rotated = _apply_gptj_rope_tail(
            normed,
            compressed_position,
            cos_sin_cache,
            DEEPSEEK_V4_ROPE_DIM,
        )
        rotated = _deepseek_v4_hadamard_rotate(rotated).float()
        writer = (
            write_deepseek_v4_indexer_mxfp4_cache
            if use_fp4_cache
            else write_deepseek_v4_indexer_fp8_cache
        )
        writer(
            rotated.unsqueeze(0),
            kv_cache_2d,
            kv_slot_mapping[token_idx : token_idx + 1],
            block_size=kv_cache_block_size,
        )
