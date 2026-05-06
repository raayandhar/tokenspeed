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

"""TRT-LLM fp8 quantize kernels exposed via the numerics registry.

We pair these against torch reference implementations so per-SM accuracy gets
exercised on every CI run. Each registered impl returns ``qweight.float()`` —
the fp8 quantized values cast back to fp32 for comparison. We deliberately
skip comparing the scale tensor: its memory layout (1D vs 2D, padded vs
contiguous) differs between SM90 and SM100 in trtllm; the qweight values
already prove the per-group statistics + rounding agree.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement, Platform
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.thirdparty.trtllm import (
    per_tensor_quant_fp8 as _trtllm_per_tensor_quant_fp8,
)
from tokenspeed_kernel.thirdparty.trtllm import (
    per_token_group_quant_8bit as _trtllm_per_token_group_quant_8bit,
)
from tokenspeed_kernel.thirdparty.trtllm import (
    per_token_quant_fp8 as _trtllm_per_token_quant_fp8,
)

_FP8_DTYPE = Platform.get().fp8e4m3fn.dtype


@register_kernel(
    "quantize",
    "fp8_token_group_128",
    name="trtllm_fp8_token_group_128",
    solution="trtllm",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    # The trtllm op asserts bf16-only input.
    dtypes={torch.bfloat16},
    traits={},
    priority=12,
    tags={"latency"},
)
def trtllm_fp8_token_group_128(x: torch.Tensor) -> torch.Tensor:
    qweight, _scale = _trtllm_per_token_group_quant_8bit(x, group_size=128)
    return qweight.float()


@register_kernel(
    "quantize",
    "fp8_token",
    name="trtllm_fp8_token",
    solution="trtllm",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.bfloat16, torch.float16},
    traits={},
    priority=12,
    tags={"latency"},
)
def trtllm_fp8_token(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.empty(x.size(0), dtype=torch.float32, device=x.device)
    _trtllm_per_token_quant_fp8(x, output, scale)
    return output.float()


@register_kernel(
    "quantize",
    "fp8_tensor",
    name="trtllm_fp8_tensor",
    solution="trtllm",
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        vendors=frozenset({"nvidia"}),
    ),
    dtypes={torch.bfloat16, torch.float16},
    traits={},
    priority=12,
    tags={"latency"},
)
def trtllm_fp8_tensor(x: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x, dtype=_FP8_DTYPE)
    scale = torch.zeros(1, dtype=torch.float32, device=x.device)
    _trtllm_per_tensor_quant_fp8(x, output, scale)
    return output.float()
