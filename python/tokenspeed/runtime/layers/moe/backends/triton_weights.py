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

from __future__ import annotations

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.base import MoEBackend
from tokenspeed.runtime.layers.moe.backends.weights import create_moe_weight_pair
from tokenspeed.runtime.utils import set_weight_attrs

__all__ = [
    "attach_dense_weight_pair",
    "register_block_scale_inverses",
    "register_input_scale_placeholders",
    "register_per_channel_scales",
]


def attach_dense_weight_pair(
    backend: MoEBackend,
    layer: nn.Module,
    *,
    with_bias: bool = False,
    params_dtype: torch.dtype,
) -> int:
    ispp = backend.spec.intermediate_size // backend.spec.tp_size
    create_moe_weight_pair(
        backend,
        layer,
        backend.spec.num_local_experts,
        backend.spec.hidden_size,
        ispp,
        params_dtype,
        with_bias=with_bias,
    )
    return ispp


def register_block_scale_inverses(
    backend,
    layer: nn.Module,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    block_shape: tuple[int, int],
) -> None:
    block_n, block_k = block_shape
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(
            num_local_experts,
            2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
            (hidden_size + block_k - 1) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(
            num_local_experts,
            (hidden_size + block_n - 1) // block_n,
            (intermediate_size_per_partition + block_k - 1) // block_k,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
    layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

    weight_loader = backend._make_weight_loader()
    set_weight_attrs(w13_weight_scale, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": weight_loader})


def register_per_channel_scales(
    backend,
    layer: nn.Module,
    *,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
) -> None:
    w13_weight_scale = torch.nn.Parameter(
        torch.ones(
            num_local_experts,
            2 * intermediate_size_per_partition,
            1,
            dtype=torch.float32,
        ),
        requires_grad=False,
    )
    w2_weight_scale = torch.nn.Parameter(
        torch.ones(num_local_experts, hidden_size, 1, dtype=torch.float32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_weight_scale)
    layer.register_parameter("w2_weight_scale", w2_weight_scale)

    scale_loader = backend._make_per_channel_scale_loader()
    set_weight_attrs(w13_weight_scale, {"weight_loader": scale_loader})
    set_weight_attrs(w2_weight_scale, {"weight_loader": scale_loader})


def register_input_scale_placeholders(layer: nn.Module) -> None:
    layer.register_parameter("w13_input_scale", None)
    layer.register_parameter("w2_input_scale", None)
