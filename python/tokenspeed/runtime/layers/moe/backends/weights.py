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

from typing import TYPE_CHECKING

import torch
from torch import nn

from tokenspeed.runtime.utils import set_weight_attrs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.moe.backends.base import MoEBackend


def create_moe_weight_pair(
    backend: MoEBackend,
    layer: nn.Module,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    params_dtype: torch.dtype,
    with_bias: bool = False,
) -> None:
    # Fused gate_up_proj (column parallel)
    w13_weight = torch.nn.Parameter(
        torch.empty(
            num_local_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )
    # down_proj (row parallel)
    w2_weight = torch.nn.Parameter(
        torch.empty(
            num_local_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype,
        ),
        requires_grad=False,
    )

    layer.register_parameter("w13_weight", w13_weight)
    layer.register_parameter("w2_weight", w2_weight)

    weight_loader = backend._make_weight_loader()
    set_weight_attrs(w13_weight, {"weight_loader": weight_loader})
    set_weight_attrs(w2_weight, {"weight_loader": weight_loader})

    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_bias = torch.nn.Parameter(
            torch.zeros(num_local_experts, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        set_weight_attrs(w13_weight_bias, {"weight_loader": weight_loader})
        set_weight_attrs(w2_weight_bias, {"weight_loader": weight_loader})
