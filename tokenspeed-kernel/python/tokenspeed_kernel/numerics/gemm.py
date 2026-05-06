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

import math
from typing import Any

import torch
from tokenspeed_kernel.numerics.inputs import (
    InputGenerator,
    set_benchmark_shapes,
    set_input_generator,
    set_standard_shapes,
)
from tokenspeed_kernel.numerics.tolerance import Tolerance, set_family_tolerance

# ---------------------------------------------------------------------------
# Tolerance
# ---------------------------------------------------------------------------

_ATOL = {
    torch.float32: 1e-5,
    # bf16/fp16: error is dominated by the output cast (~1 ulp_rel = 2^-7 ≈ 8e-3
    # for bf16; 2^-10 ≈ 1e-3 for fp16), not by fp32 accumulation, so we set the
    # baseline at the rounding floor and use a K-independent scale.
    torch.float16: 1.5e-2,
    torch.bfloat16: 1.5e-2,
    torch.float8_e4m3fn: 5e-3,
    torch.float8_e4m3fnuz: 5e-3,
}

_FP8_DTYPES: set[torch.dtype] = {
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
}

_BF16_FP16_DTYPES: set[torch.dtype] = {
    torch.float16,
    torch.bfloat16,
}


def tolerance(
    dtype: torch.dtype,
    *,
    K: int | None = None,
    inputs: dict[str, Any] | None = None,
    acc_dtype: torch.dtype = torch.float32,
    **_: Any,
) -> Tolerance:
    """Shape-aware GEMM tolerance.

    - fp32: error grows as sqrt(K) under fp32 accumulation noise.
    - fp16/bf16: K-independent — fp32 accumulation is well below the output
      dtype's rounding floor, so error is dominated by the final cast.
    - fp8: error grows linearly with K for blockwise kernels.
    """
    if dtype not in _ATOL:
        raise KeyError(f"No GEMM tolerance baseline for dtype={dtype}")

    if K is None and inputs is not None and "A" in inputs:
        K = int(inputs["A"].shape[-1])
    if K is None:
        raise ValueError("GEMM tolerance requires K or inputs['A']")

    base = _ATOL[dtype]
    if dtype in _FP8_DTYPES:
        scale = max(K, 1) / 128.0
    elif dtype in _BF16_FP16_DTYPES:
        scale = 1.0
    else:
        scale = math.sqrt(max(K, 1) / 128.0)
    if acc_dtype != torch.float32:
        scale *= 8.0
    return Tolerance(atol=base * scale, rtol=base * scale)


set_family_tolerance("gemm", tolerance)

# ---------------------------------------------------------------------------
# Input Generator
# ---------------------------------------------------------------------------


class GemmInputGenerator(InputGenerator):
    def _generate_value(self, shape: tuple[int, ...], dtype) -> torch.Tensor:
        values = torch.randn(
            *shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.rng,
        )
        return values.to(dtype)

    def _generate_scales(self, shape: tuple[int, ...], dtype) -> torch.Tensor:
        scales = torch.rand(
            *shape,
            dtype=torch.float32,
            device=self.device,
            generator=self.rng,
        )
        return scales.to(dtype)

    def generate(
        self,
        M: int,
        N: int,
        K: int,
    ) -> dict[str, Any]:
        quant = self.traits.get("quant")
        scale_type = self.traits.get("scale_type")
        a_layout = self.traits.get("a_layout")
        b_layout = self.traits.get("b_layout")

        A = (
            self._generate_value((K, M), self.dtype)
            if a_layout == {"KM"}
            else self._generate_value((M, K), self.dtype)
        )
        B = (
            self._generate_value((K, N), self.dtype)
            if b_layout == {"KN"}
            else self._generate_value((N, K), self.dtype)
        )

        A_scales = None
        B_scales = None
        block_size = None

        if quant == {"mxfp8"}:
            block_size = [128, 128]
            k_tiles = math.ceil(K / block_size[0])
            n_tiles = math.ceil(N / block_size[1])
            A_scales = self._generate_scales((M, k_tiles), torch.float32)
            B_scales = self._generate_scales((n_tiles, k_tiles), torch.float32)
        else:
            if scale_type == {"per_channel"}:
                A_scales = self._generate_scales((M,), torch.float32)
                B_scales = self._generate_scales((N,), torch.float32)
            else:
                A_scales = self._generate_scales((1,), torch.float32)
                B_scales = self._generate_scales((1,), torch.float32)

        out_dtype = torch.bfloat16
        alpha = None

        return {
            "A": A,
            "B": B,
            "A_scales": A_scales,
            "B_scales": B_scales,
            "out_dtype": out_dtype,
            "alpha": alpha,
            "block_size": block_size,
        }


set_input_generator("gemm", "mm", GemmInputGenerator)

# ---------------------------------------------------------------------------
# Shape Presets
# ---------------------------------------------------------------------------


GEMM_MM_STANDARD_SHAPES: list[dict[str, int]] = [
    {"M": 16, "N": 16, "K": 64},
    {"M": 64, "N": 128, "K": 128},
    {"M": 128, "N": 128, "K": 256},
    {"M": 256, "N": 256, "K": 512},
    # DSv3 hot-path shapes — exercise hand-rolled kernels in trtllm dsv3_router /
    # dsv3_fused_a (M ≤ 16, K = 7168, N = num_experts=256 or fused_a=2112) plus
    # off-shape (M = 64) which falls back to cuBLAS inside the same op.
    {"M": 1, "N": 256, "K": 7168},
    {"M": 8, "N": 256, "K": 7168},
    {"M": 16, "N": 256, "K": 7168},
    {"M": 64, "N": 256, "K": 7168},
    {"M": 1, "N": 2112, "K": 7168},
    {"M": 8, "N": 2112, "K": 7168},
    {"M": 16, "N": 2112, "K": 7168},
    {"M": 64, "N": 2112, "K": 7168},
]


GEMM_MM_BENCHMARK_SHAPES: list[dict[str, int]] = [
    {"M": 1, "N": 4096, "K": 4096},
    {"M": 16, "N": 4096, "K": 4096},
    {"M": 128, "N": 4096, "K": 4096},
    {"M": 512, "N": 4096, "K": 4096},
    {"M": 4096, "N": 4096, "K": 4096},
]


set_standard_shapes("gemm", "mm", GEMM_MM_STANDARD_SHAPES)
set_benchmark_shapes("gemm", "mm", GEMM_MM_BENCHMARK_SHAPES)
