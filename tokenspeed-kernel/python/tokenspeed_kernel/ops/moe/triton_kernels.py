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

"""Registration of triton_kernels-based MoE kernels.

The actual implementations live in tokenspeed_kernel.thirdparty.triton_kernels.
This module imports and registers them so they are discoverable via
select_kernel("moe", ...).
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.registry import Priority, register_kernel

try:
    import tokenspeed_kernel.thirdparty.triton_kernels.matmul_ogs_details.opt_flags as opt_flags
except ImportError:
    opt_flags = None

try:
    from tokenspeed_kernel.thirdparty.triton_kernels.matmul_ogs import (
        FlexCtx,
        FnSpecs,
        FusedActivation,
        PrecisionConfig,
    )
    from tokenspeed_kernel.thirdparty.triton_kernels.numerics import InFlexData
    from tokenspeed_kernel.thirdparty.triton_kernels.swiglu import swiglu_fn
    from tokenspeed_kernel.thirdparty.triton_kernels.tensor import (
        FP4,
        convert_layout,
        wrap_torch_tensor,
    )
    from tokenspeed_kernel.thirdparty.triton_kernels.tensor_details import layout
except ImportError:
    FlexCtx = None
    FnSpecs = None
    FusedActivation = None
    PrecisionConfig = None
    InFlexData = None
    swiglu_fn = None
    FP4 = None
    convert_layout = None
    wrap_torch_tensor = None
    layout = None

# --- triton_kernels routing (softmax + top-k) ---
try:
    from tokenspeed_kernel.thirdparty.triton_kernels.routing import routing

    routing = register_kernel(
        "moe",
        "route",
        name="triton_kernels_routing",
        solution="triton",
        dtypes={torch.float16, torch.bfloat16, torch.float32},
        traits={"output_type": frozenset({"routing_data"})},
        priority=Priority.PERFORMANT + 2,
        tags={"portability"},
    )(routing)
except ImportError:
    pass

# --- triton_kernels matmul_ogs (MoE expert GEMM with gather/scatter) ---
#
# matmul_ogs supports three modes depending on parameters:
#   dispatch_gemm  – gather/dispatch tokens then GEMM (gate_up projection)
#   gemm_combine   – GEMM then scatter/combine results (down projection)
# We register it three times so each mode can be selected independently.
try:
    from tokenspeed_kernel.thirdparty.triton_kernels.matmul_ogs import matmul_ogs

    _matmul_ogs_common = dict(
        solution="triton",
        dtypes={torch.float16, torch.bfloat16, torch.uint8},
        priority=Priority.PERFORMANT + 2,
        tags={"portability"},
    )

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_matmul_ogs",
        features={"routing_data"},
        **_matmul_ogs_common,
    )(matmul_ogs)

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_dispatch_gemm",
        features={"routing_data", "dispatch_gemm"},
        **_matmul_ogs_common,
    )(matmul_ogs)

    register_kernel(
        "moe",
        "experts",
        name="triton_kernels_gemm_combine",
        features={"routing_data", "gemm_combine"},
        **_matmul_ogs_common,
    )(matmul_ogs)
except ImportError:
    pass
