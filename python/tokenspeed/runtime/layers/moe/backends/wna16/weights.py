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

import enum
from enum import Enum
from functools import partial

import torch
from torch import nn

from tokenspeed.runtime.layers.moe.backends.wna16.marlin_weight_loaders import (
    load_model_weight,
)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


def attach_marlin_weights(backend, layer: nn.Module) -> None:
    from tokenspeed.runtime.utils import set_weight_attrs

    ispp = backend.spec.intermediate_size // backend.spec.tp_size
    num_experts = backend.spec.num_local_experts
    hidden = backend.spec.hidden_size
    tp_rank = backend.spec.tp_rank
    params_dtype = torch.get_default_dtype()
    packed_factor = backend._packed_factor

    # Will transpose the loaded weight along the
    # intermediate and hidden dim sizes. Will
    # shard for TP along the transposed dims
    extra_weight_attrs = {"is_transposed": True, "quant_method": backend._strategy}

    # Fused gate_up_proj (column parallel)
    w13_weight = torch.nn.Parameter(
        torch.empty(num_experts, hidden // packed_factor, 2 * ispp, dtype=torch.int32),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_packed", w13_weight)
    set_weight_attrs(w13_weight, extra_weight_attrs)
    set_weight_attrs(
        w13_weight,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w13_weight_packed", tp_rank=tp_rank
            )
        },
    )

    # down_proj (row parallel)
    w2_weight = torch.nn.Parameter(
        torch.empty(num_experts, ispp // packed_factor, hidden, dtype=torch.int32),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_packed", w2_weight)
    set_weight_attrs(w2_weight, extra_weight_attrs)
    set_weight_attrs(
        w2_weight,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w2_weight_packed", tp_rank=tp_rank
            )
        },
    )

    # In the case where we have actorder/g_idx,
    # we do not partition the w2 scales
    load_full_w2 = backend._actorder and backend._group_size != -1
    backend._is_k_full = (not backend._actorder) or backend.spec.tp_size == 1
    w2_scales_size = ispp * backend.spec.tp_size if load_full_w2 else ispp

    if backend._strategy == "channel":
        num_groups_w2 = num_groups_w13 = 1
        backend._group_size = -1
    else:
        num_groups_w2 = w2_scales_size // backend._group_size
        num_groups_w13 = hidden // backend._group_size

    w13_scale = torch.nn.Parameter(
        torch.ones(num_experts, num_groups_w13, 2 * ispp, dtype=params_dtype),
        requires_grad=False,
    )
    layer.register_parameter("w13_weight_scale", w13_scale)
    set_weight_attrs(w13_scale, extra_weight_attrs)
    set_weight_attrs(
        w13_scale,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w13_weight_scale", tp_rank=tp_rank
            )
        },
    )

    w2_scale = torch.nn.Parameter(
        torch.ones(num_experts, num_groups_w2, hidden, dtype=params_dtype),
        requires_grad=False,
    )
    layer.register_parameter("w2_weight_scale", w2_scale)
    set_weight_attrs(w2_scale, extra_weight_attrs)
    set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})
    set_weight_attrs(
        w2_scale,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w2_weight_scale", tp_rank=tp_rank
            )
        },
    )

    w2_weight_shape = torch.nn.Parameter(
        torch.empty(num_experts, 2), requires_grad=False
    )
    layer.register_parameter("w2_weight_shape", w2_weight_shape)
    set_weight_attrs(w2_weight_shape, extra_weight_attrs)
    set_weight_attrs(
        w2_weight_shape,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w2_weight_shape", tp_rank=tp_rank
            )
        },
    )

    w13_weight_shape = torch.nn.Parameter(
        torch.empty(num_experts, 2), requires_grad=False
    )
    layer.register_parameter("w13_weight_shape", w13_weight_shape)
    set_weight_attrs(w13_weight_shape, extra_weight_attrs)
    set_weight_attrs(
        w13_weight_shape,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w13_weight_shape", tp_rank=tp_rank
            )
        },
    )

    w13_g_idx = torch.nn.Parameter(
        torch.empty(num_experts, hidden, dtype=torch.int32), requires_grad=False
    )
    layer.register_parameter("w13_weight_g_idx", w13_g_idx)
    set_weight_attrs(w13_g_idx, extra_weight_attrs)
    set_weight_attrs(
        w13_g_idx,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w13_weight_g_idx", tp_rank=tp_rank
            )
        },
    )

    w2_g_idx = torch.nn.Parameter(
        torch.empty(num_experts, ispp, dtype=torch.int32), requires_grad=False
    )
    layer.register_parameter("w2_weight_g_idx", w2_g_idx)
    set_weight_attrs(w2_g_idx, extra_weight_attrs)
    set_weight_attrs(
        w2_g_idx,
        {
            "weight_loader": partial(
                load_model_weight, weight_name="w2_weight_g_idx", tp_rank=tp_rank
            )
        },
    )

    w13_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty(num_experts, hidden, dtype=torch.int32), requires_grad=False
    )
    layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
    set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)
    set_weight_attrs(
        w13_g_idx_sort_indices,
        {
            "weight_loader": partial(
                load_model_weight,
                weight_name="w13_g_idx_sort_indices",
                tp_rank=tp_rank,
            )
        },
    )

    w2_g_idx_sort_indices = torch.nn.Parameter(
        torch.empty(num_experts, ispp, dtype=torch.int32), requires_grad=False
    )
    layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
    set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)
    set_weight_attrs(
        w2_g_idx_sort_indices,
        {
            "weight_loader": partial(
                load_model_weight,
                weight_name="w2_g_idx_sort_indices",
                tp_rank=tp_rank,
            )
        },
    )

    layer.a13_scale = None
    layer.a2_scale = None
    layer.marlin_state = GPTQMarlinState.REPACK
