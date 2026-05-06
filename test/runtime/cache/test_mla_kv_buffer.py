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

import pytest
import torch

from tokenspeed.runtime.cache.utils import (
    get_mla_kv_buffer_triton,
    set_mla_kv_buffer_triton,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# K2.5 / DSv3 MLA dims.
NOPE_DIM = 512
ROPE_DIM = 64
TOTAL_DIM = NOPE_DIM + ROPE_DIM
NUM_PAGES = 50_000

# Spans both dispatch branches (small n -> block-split, large n -> per-loc).
N_LOC_SMALL = [4, 64, 128, 256, 511]
N_LOC_LARGE = [512, 1024, 4096]
N_LOC_ALL = N_LOC_SMALL + N_LOC_LARGE


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _make_inputs(n_loc: int, dtype: torch.dtype, pattern: str, seed: int = 0):
    torch.manual_seed(seed)
    device = "cuda"
    if pattern == "seq":
        loc = torch.arange(n_loc, device=device, dtype=torch.int64)
    else:
        loc = torch.randperm(NUM_PAGES, device=device, dtype=torch.int64)[:n_loc]

    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(n_loc, 1, NOPE_DIM, device=device, dtype=torch.bfloat16) * 50
        k_nope = bf.to(dtype)
        bf = torch.randn(n_loc, 1, ROPE_DIM, device=device, dtype=torch.bfloat16) * 50
        k_rope = bf.to(dtype)
    else:
        k_nope = torch.randn(n_loc, 1, NOPE_DIM, device=device, dtype=dtype)
        k_rope = torch.randn(n_loc, 1, ROPE_DIM, device=device, dtype=dtype)
    return loc, k_nope, k_rope


def _empty_kv(dtype: torch.dtype) -> torch.Tensor:
    """Allocate an unused-cell sentinel-filled kv_buffer so untouched cells
    diverge if the kernel writes to them."""
    sentinel = torch.full(
        (NUM_PAGES, TOTAL_DIM), 7.5, device="cuda", dtype=torch.bfloat16
    )
    return sentinel.to(dtype) if dtype == torch.float8_e4m3fn else sentinel.to(dtype)


def _torch_set_reference(kv: torch.Tensor, loc, k_nope, k_rope) -> torch.Tensor:
    """Pure-torch scatter-write reference."""
    out = kv.clone()
    out[loc, :NOPE_DIM] = k_nope[:, 0, :]
    out[loc, NOPE_DIM:] = k_rope[:, 0, :]
    return out


def _torch_get_reference(kv: torch.Tensor, loc) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch scatter-read reference."""
    return (
        kv[loc, :NOPE_DIM].unsqueeze(1).contiguous(),
        kv[loc, NOPE_DIM:].unsqueeze(1).contiguous(),
    )


# ─── set ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", N_LOC_ALL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_set_matches_torch_reference(n_loc, dtype, pattern):
    """set_mla_kv_buffer_triton scatters k_nope/k_rope into kv_buffer at loc
    indices, byte-for-byte vs a torch reference. Spans both dispatch branches
    via the n_loc parametrization."""
    loc, k_nope, k_rope = _make_inputs(n_loc, dtype, pattern)
    kv = _empty_kv(dtype)
    ref = _torch_set_reference(kv, loc, k_nope, k_rope)

    set_mla_kv_buffer_triton(kv, loc, k_nope, k_rope)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv, ref)


@pytest.mark.parametrize("n_loc", [4, 511, 512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_set_pdl_invariant(n_loc, dtype):
    """PDL is a scheduling hint; output must be bitwise-identical regardless."""
    loc, k_nope, k_rope = _make_inputs(n_loc, dtype, "rand")
    kv_off = _empty_kv(dtype)
    kv_on = _empty_kv(dtype)

    set_mla_kv_buffer_triton(kv_off, loc, k_nope, k_rope, enable_pdl=False)
    set_mla_kv_buffer_triton(kv_on, loc, k_nope, k_rope, enable_pdl=True)
    torch.cuda.synchronize()

    assert _bitwise_equal(kv_off, kv_on)


# ─── get ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", N_LOC_ALL)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("pattern", ["seq", "rand"])
def test_get_matches_torch_reference(n_loc, dtype, pattern):
    """get_mla_kv_buffer_triton gathers from kv_buffer at loc indices into
    cache_k_nope / cache_k_rope outputs, byte-for-byte vs a torch reference."""
    # Populate kv_buffer with random data we'll read back.
    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=torch.bfloat16) * 50
        kv = bf.to(dtype)
    else:
        kv = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=dtype)

    if pattern == "seq":
        loc = torch.arange(n_loc, device="cuda", dtype=torch.int64)
    else:
        loc = torch.randperm(NUM_PAGES, device="cuda", dtype=torch.int64)[:n_loc]

    k_nope = torch.empty((n_loc, 1, NOPE_DIM), dtype=dtype, device="cuda")
    k_rope = torch.empty((n_loc, 1, ROPE_DIM), dtype=dtype, device="cuda")
    nope_ref, rope_ref = _torch_get_reference(kv, loc)

    get_mla_kv_buffer_triton(kv, loc, k_nope, k_rope)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope, nope_ref)
    assert _bitwise_equal(k_rope, rope_ref)


@pytest.mark.parametrize("n_loc", [4, 511, 512, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_get_pdl_invariant(n_loc, dtype):
    if dtype == torch.float8_e4m3fn:
        bf = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=torch.bfloat16) * 50
        kv = bf.to(dtype)
    else:
        kv = torch.randn(NUM_PAGES, TOTAL_DIM, device="cuda", dtype=dtype)

    loc = torch.randperm(NUM_PAGES, device="cuda", dtype=torch.int64)[:n_loc]

    k_nope_off = torch.empty((n_loc, 1, NOPE_DIM), dtype=dtype, device="cuda")
    k_rope_off = torch.empty((n_loc, 1, ROPE_DIM), dtype=dtype, device="cuda")
    k_nope_on = torch.empty_like(k_nope_off)
    k_rope_on = torch.empty_like(k_rope_off)

    get_mla_kv_buffer_triton(kv, loc, k_nope_off, k_rope_off, enable_pdl=False)
    get_mla_kv_buffer_triton(kv, loc, k_nope_on, k_rope_on, enable_pdl=True)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope_off, k_nope_on)
    assert _bitwise_equal(k_rope_off, k_rope_on)


# ─── round trip ─────────────────────────────────────────────────────


@pytest.mark.parametrize("n_loc", [128, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_set_then_get_round_trip(n_loc, dtype):
    """set followed by get on the same loc indices recovers the original
    k_nope / k_rope inputs bitwise."""
    loc, k_nope_in, k_rope_in = _make_inputs(n_loc, dtype, "rand")
    kv = _empty_kv(dtype)

    set_mla_kv_buffer_triton(kv, loc, k_nope_in, k_rope_in)

    k_nope_out = torch.empty_like(k_nope_in)
    k_rope_out = torch.empty_like(k_rope_in)
    get_mla_kv_buffer_triton(kv, loc, k_nope_out, k_rope_out)
    torch.cuda.synchronize()

    assert _bitwise_equal(k_nope_out, k_nope_in)
    assert _bitwise_equal(k_rope_out, k_rope_in)
