// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "flashinfer/routing_flash.cuh"
#include "flashinfer/utils.cuh"
#include "tvm_ffi_utils.h"

using namespace flashinfer::routing_flash;

int get_sm_count() {
  static int sm_count = 0;
  if (sm_count == 0) {
    int device_id;
    FLASHINFER_CUDA_CALL(cudaGetDevice(&device_id));
    FLASHINFER_CUDA_CALL(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
  }
  return sm_count;
}

void softmax_topk_flash(TensorView input, TensorView correction_bias, TensorView topk_indices,
                        TensorView topk_weights, int64_t num_experts_real, float scaling_factor,
                        bool renormalize) {
  TVM_FFI_ICHECK_EQ(topk_weights.dtype(), dl_float32);
  const int num_experts = input.size(1);
  const int total_num_tokens = input.size(0);
  const int topk = topk_weights.size(1);
  const cudaStream_t stream = get_stream(input.device());

#define NUM_EXPERTS_SWITCH(NUM_EXPERTS_, ...)                                                 \
  [&] {                                                                                       \
    if (NUM_EXPERTS_ == 384) {                                                                \
      constexpr static int NUM_EXPERTS = 384;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 576) {                                                         \
      constexpr static int NUM_EXPERTS = 576;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 768) {                                                         \
      constexpr static int NUM_EXPERTS = 768;                                                 \
      return __VA_ARGS__();                                                                   \
    } else if (NUM_EXPERTS_ == 896) {                                                         \
      constexpr static int NUM_EXPERTS = 896;                                                 \
      return __VA_ARGS__();                                                                   \
    } else {                                                                                  \
      throw std::runtime_error("Not supported num experts: " + std::to_string(NUM_EXPERTS_)); \
    }                                                                                         \
  }()

#define IDTYPE_SWITCH(DTYPE_CODE, IDTYPE, ...)                                    \
  [&] {                                                                           \
    if (DTYPE_CODE == int64_code) {                                               \
      using IDTYPE = int64_t;                                                     \
      return __VA_ARGS__();                                                       \
    } else if (DTYPE_CODE == int32_code) {                                        \
      using IDTYPE = int32_t;                                                     \
      return __VA_ARGS__();                                                       \
    } else {                                                                      \
      TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported indices dtype."; \
    }                                                                             \
  }()

  NUM_EXPERTS_SWITCH(num_experts, [&] {
    TVM_FFI_ICHECK(NUM_EXPERTS > num_experts_real);
    // Single Warp
    cudaLaunchConfig_t config;
    config.gridDim = min(max(total_num_tokens, 1), 2048);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = true;
    config.numAttrs = 1;
    config.attrs = attrs;
    int64_t indices_dtype_code = encode_dlpack_dtype(topk_indices.dtype());

    IDTYPE_SWITCH(indices_dtype_code, IndexT, [&] {
      if constexpr (NUM_EXPERTS == 576) {
        static constexpr int vec_size = 4;
        TVM_FFI_ICHECK(NUM_EXPERTS % vec_size == 0);
        TVM_FFI_ICHECK(vec_size % 4 == 0);
        TVM_FFI_ICHECK(topk % 4 == 0);

        static constexpr int block_size = NUM_EXPERTS / vec_size;
        config.blockDim = block_size;
        auto kernel =
            flashinfer::routing_flash::softmax_topk_correction_bias_zero_experts_fuse_kernel<
                vec_size, block_size, IndexT>;

        cudaLaunchKernelEx(&config, kernel, static_cast<float*>(input.data_ptr()),
                          static_cast<float*>(correction_bias.data_ptr()), static_cast<IndexT*>(topk_indices.data_ptr()),
                          static_cast<float*>(topk_weights.data_ptr()), topk, total_num_tokens,
                          num_experts, static_cast<int>(num_experts_real),
                          static_cast<float>(scaling_factor), renormalize);
      } else {
        static constexpr int vec_size = (NUM_EXPERTS / 32);
        TVM_FFI_ICHECK(NUM_EXPERTS % vec_size == 0);
        TVM_FFI_ICHECK(vec_size % 4 == 0);

        static constexpr int block_size = 32;
        config.blockDim = block_size;
        auto kernel =
            flashinfer::routing_flash::softmax_topk_correction_bias_zero_experts_fuse_kernel_single_warp<
                vec_size, block_size, IndexT>;

        cudaLaunchKernelEx(&config, kernel, static_cast<float*>(input.data_ptr()),
                          static_cast<float*>(correction_bias.data_ptr()), static_cast<IndexT*>(topk_indices.data_ptr()),
                          static_cast<float*>(topk_weights.data_ptr()), topk, total_num_tokens,
                          num_experts, static_cast<int>(num_experts_real),
                          static_cast<float>(scaling_factor), renormalize);
      }
    });
    cudaError_t err = cudaGetLastError();
    TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);
    return true;
  });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(softmax_topk_flash, softmax_topk_flash);
