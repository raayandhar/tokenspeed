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

"""Helpers shared across runtime model implementations."""

import torch
from tokenspeed_kernel.ops.embedding import FusedSetKVBufferArg
from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.layers.paged_attention import PagedAttention
from tokenspeed.runtime.utils import print_warning_once


def create_fused_set_kv_buffer_arg(
    value: torch.Tensor,
    layer: PagedAttention,
    out_cache_loc: torch.Tensor,
    token_to_kv_pool,
):
    """Build fused RoPE+KV write arguments when the fused path is supported."""

    from tokenspeed.runtime.layers.attention.kv_cache.mla import MLATokenToKVPool

    layer_id = layer.layer_id

    k_buffer = token_to_kv_pool.get_key_buffer(layer_id)
    v_buffer = token_to_kv_pool.get_value_buffer(layer_id)

    is_mla = isinstance(token_to_kv_pool, MLATokenToKVPool)

    if is_mla:
        kv_lora_rank = token_to_kv_pool.kv_lora_rank
        k_buffer = k_buffer[..., kv_lora_rank:].view(k_buffer.shape[0], -1)
        v_buffer = v_buffer[..., :kv_lora_rank].view(v_buffer.shape[0], -1)
    else:
        k_buffer = k_buffer.view(k_buffer.shape[0], -1)
        v_buffer = v_buffer.view(v_buffer.shape[0], -1)

    # Non-trivial scales need 1/scale applied before FP8 cast — the fused kernel
    # doesn't support this yet, so log a warning and skip the fused path.
    k_scale = layer.k_scale
    v_scale = layer.v_scale
    if (k_scale is not None and k_scale != 1.0) or (
        v_scale is not None and v_scale != 1.0
    ):
        print_warning_once(
            f"Fused RoPE+KV write disabled: non-trivial k_scale={k_scale} v_scale={v_scale}"
        )
        return None

    return FusedSetKVBufferArg(
        value=value,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_scale=None,
        v_scale=None,
        cache_loc=out_cache_loc,
    )
