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

"""Reference MoE align_block_size implementation."""

from __future__ import annotations

import torch
from tokenspeed_kernel.numerics.moe import (
    canonicalize_align_block_size,
    compute_align_block_size_buffer_dims,
)
from tokenspeed_kernel.registry import register_kernel


@register_kernel(
    "moe",
    "align_block_size",
    name="torch_moe_align_block_size",
    solution="reference",
    dtypes={torch.int32},
    traits={},
    priority=10,
    tags={"determinism", "portability"},
)
def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> torch.Tensor:
    """Pure-torch reference for moe_align_block_size.

    Returns the canonical packed tensor (see ``canonicalize_align_block_size``).
    """
    assert topk_ids.dtype == torch.int32
    total_tokens, top_k = topk_ids.shape
    device = topk_ids.device
    pad_id = total_tokens * top_k

    max_num_m_blocks, sorted_ids_size = compute_align_block_size_buffer_dims(
        pad_id, num_experts, block_size
    )

    # For each (token, k) slot, assign a flat-id 0..pad_id-1; group by expert.
    flat_token_ids = torch.arange(pad_id, device=device, dtype=torch.int32)
    flat_expert = topk_ids.flatten()

    # Gather slot ids per expert (e in [0, num_experts) is a real expert; e =
    # num_experts marks the filtered-out / "no expert" slot in EP setups).
    sorted_ids = torch.full(
        (sorted_ids_size,), pad_id, device=device, dtype=torch.int32
    )
    expert_ids = torch.zeros((max_num_m_blocks,), device=device, dtype=torch.int32)

    write_pos = 0
    block_idx = 0
    for e in range(num_experts + 1):
        slot_ids = flat_token_ids[flat_expert == e]
        n_slots = slot_ids.numel()
        # Pad to multiple of block_size.
        n_blocks = (n_slots + block_size - 1) // block_size
        for b in range(n_blocks):
            block_start = write_pos + b * block_size
            block_slot_count = min(block_size, n_slots - b * block_size)
            sorted_ids[block_start : block_start + block_slot_count] = slot_ids[
                b * block_size : b * block_size + block_slot_count
            ]
            expert_ids[block_idx + b] = e
        write_pos += n_blocks * block_size
        block_idx += n_blocks

    num_tokens_post_pad = torch.tensor([write_pos], device=device, dtype=torch.int32)
    return canonicalize_align_block_size(
        sorted_ids, expert_ids, num_tokens_post_pad, block_size
    )
