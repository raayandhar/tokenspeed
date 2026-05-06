/*
 * Copyright (c) 2026 LightSeek Foundation
 *
 * Vendored from flashinfer/attention/cascade.cuh (Apache-2.0):
 *   https://github.com/flashinfer-ai/flashinfer
 * with two additions:
 *   1. ``lse_scale_log2`` / ``lse_scale_inv`` runtime args so callers can pass
 *      LSE in any base; the kernel rebases to log2 internally (PTX-native
 *      ``ex2.approx``) and rebases back at store. Default callers pass natural
 *      log (LN→log2 via lse_scale_log2 = log2(e)); flashinfer-style log2 callers
 *      pass lse_scale_log2 = 1.0.
 *   2. PDL via ``griddepcontrol.{wait,launch_dependents}`` + the matching
 *      cudaLaunchAttributeProgrammaticStreamSerialization attribute, so
 *      producer/consumer pairs (FA chunked attention upstream, next-iter
 *      attention or o_proj GEMM downstream) can overlap their preambles.
 */

#include <flashinfer/attention/state.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/vec_dtypes.cuh>

#include "tvm_ffi_utils.h"

namespace tokenspeed {

using flashinfer::vec_t;
namespace math = flashinfer::math;

template <uint32_t vec_size, typename DTypeIn, typename DTypeO>
__global__ void MergeStateKernel(DTypeIn* __restrict__ v_a, float* __restrict__ s_a,
                                 DTypeIn* __restrict__ v_b, float* __restrict__ s_b,
                                 DTypeO* __restrict__ v_merged, float* __restrict__ s_merged,
                                 uint32_t num_heads, uint32_t head_dim,
                                 float lse_scale_log2, float lse_scale_inv) {
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif
  uint32_t tx = threadIdx.x, ty = threadIdx.y;
  uint32_t pos = blockIdx.x;
  uint32_t head_idx = ty;

  float s_a_val = s_a[pos * num_heads + head_idx] * lse_scale_log2;
  float s_b_val = s_b[pos * num_heads + head_idx] * lse_scale_log2;
  float s_max = max(s_a_val, s_b_val);
  s_a_val = math::ptx_exp2(s_a_val - s_max);
  s_b_val = math::ptx_exp2(s_b_val - s_max);
  float a_scale = s_a_val / (s_a_val + s_b_val);
  float b_scale = s_b_val / (s_a_val + s_b_val);
  vec_t<float, vec_size> v_a_vec, v_b_vec, v_merged_vec;
  v_a_vec.cast_load(v_a + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  v_b_vec.cast_load(v_b + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
#pragma unroll
  for (uint32_t i = 0; i < vec_size; ++i) {
    v_merged_vec[i] = a_scale * v_a_vec[i] + b_scale * v_b_vec[i];
  }
  v_merged_vec.cast_store(v_merged + (pos * num_heads + head_idx) * head_dim + tx * vec_size);
  s_merged[pos * num_heads + head_idx] =
      (math::ptx_log2(s_a_val + s_b_val) + s_max) * lse_scale_inv;
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename DTypeIn, typename DTypeO>
cudaError_t MergeState(DTypeIn* v_a, float* s_a, DTypeIn* v_b, float* s_b, DTypeO* v_merged,
                       float* s_merged, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim,
                       float lse_scale_log2, float lse_scale_inv, bool enable_pdl,
                       cudaStream_t stream = nullptr) {
  DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
    constexpr uint32_t vec_size = std::max(16U / sizeof(DTypeIn), HEAD_DIM / 32U);
    uint32_t bdx = HEAD_DIM / vec_size;
    uint32_t bdy = num_heads;
    dim3 nblks(seq_len);
    dim3 nthrs(bdx, bdy);
    auto kernel = MergeStateKernel<vec_size, DTypeIn, DTypeO>;

    cudaLaunchConfig_t config;
    config.gridDim = nblks;
    config.blockDim = nthrs;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
    config.numAttrs = 1;
    config.attrs = attrs;

    FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, kernel, v_a, s_a, v_b, s_b, v_merged, s_merged,
                                             num_heads, head_dim, lse_scale_log2, lse_scale_inv));
  });
  return cudaSuccess;
}

}  // namespace tokenspeed

void merge_state(TensorView v_a, TensorView s_a, TensorView v_b, TensorView s_b,
                 TensorView v_merged, TensorView s_merged, double lse_scale_log2,
                 bool enable_pdl) {
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  CHECK_INPUT(v_merged);
  CHECK_INPUT(s_merged);
  CHECK_DEVICE(s_a, v_a);
  CHECK_DEVICE(v_b, v_a);
  CHECK_DEVICE(s_b, v_a);
  CHECK_DEVICE(v_merged, v_a);
  CHECK_DEVICE(s_merged, v_a);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_DIM(3, v_merged);
  CHECK_DIM(2, s_merged);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(v_a, v_merged);
  CHECK_SHAPE(s_a, s_b);
  CHECK_SHAPE(s_a, s_merged);
  TVM_FFI_ICHECK_EQ(v_a.size(0), s_a.size(0));
  TVM_FFI_ICHECK_EQ(v_a.size(1), s_a.size(1));

  unsigned int seq_len = v_a.size(0);
  unsigned int num_heads = v_a.size(1);
  unsigned int head_dim = v_a.size(2);

  cudaSetDevice(v_a.device().device_id);
  const cudaStream_t stream = get_stream(v_a.device());

  bool success = DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(v_a.dtype(), c_type, [&] {
    cudaError_t status = tokenspeed::MergeState(
        static_cast<c_type*>(v_a.data_ptr()), static_cast<float*>(s_a.data_ptr()),
        static_cast<c_type*>(v_b.data_ptr()), static_cast<float*>(s_b.data_ptr()),
        static_cast<c_type*>(v_merged.data_ptr()), static_cast<float*>(s_merged.data_ptr()),
        seq_len, num_heads, head_dim, static_cast<float>(lse_scale_log2),
        static_cast<float>(1.0 / lse_scale_log2), enable_pdl, stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "merge_state launch failed: " << cudaGetErrorString(status);
    return true;
  });
  TVM_FFI_ICHECK(success) << "merge_state launch failed: unsupported dtype.";
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(merge_state, merge_state);
