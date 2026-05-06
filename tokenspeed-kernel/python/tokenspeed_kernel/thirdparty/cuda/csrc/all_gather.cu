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

#include <assert.h>
#include <string>
#include <algorithm>

#include "flashinfer/comm/all_gather.cuh"
#include "flashinfer/utils.cuh"
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

using namespace flashinfer::simple_all_gather;

#define DISPATCH_NRANKS(nranks, ...) [&] {                                  \
    switch (nranks) {                                                       \
    case 2: { constexpr static int NRanks = 2; return __VA_ARGS__(); }      \
    case 4: { constexpr static int NRanks = 4; return __VA_ARGS__(); }      \
    case 8: { constexpr static int NRanks = 8; return __VA_ARGS__(); }      \
    default:                                                                \
        TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported ranks: " << nranks; \
    }                                                                       \
    }()

#define DISPATCH_FLOATING_TYPES(dtype, c_type, ...)                           \
  [&] {                                                                       \
    switch (encode_dlpack_dtype(dtype)) {                                     \
      case float16_code: { using c_type = half; return __VA_ARGS__(); }       \
      case bfloat16_code: { using c_type = __nv_bfloat16; return __VA_ARGS__(); }\
      default:                                                                \
        TVM_FFI_LOG_AND_THROW(NotImplementedError) << "Unsupported dtype";    \
    }                                                                         \
  }()

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

void simple_all_gather(TensorView allgather_in, int64_t world_size, int64_t world_rank,
    int64_t token_num, int64_t hidden_size, TensorView workspace_ptrs,
    bool launch_with_pdl, bool trigger_completion_at_end, int max_num_tokens,
    Optional<TensorView> allgather_out, Optional<int64_t> max_sm_to_use) {

    cudaSetDevice(allgather_in.device().device_id);

    DISPATCH_FLOATING_TYPES(allgather_in.dtype(), c_type, [&] {
        DISPATCH_NRANKS(world_size, [&] {
            AllGatherParams<c_type> params;
            params.nranks = world_size;
            params.rank = world_rank;
            params.comm_size = token_num * hidden_size;
            params.hidden_dim_per_rank = hidden_size;
            params.hidden_dim = world_size * hidden_size;
            params.workspace = reinterpret_cast<void**>(workspace_ptrs.data_ptr());
            params.num_tokens = token_num;
            params.allgather_in = reinterpret_cast<void*>(allgather_in.data_ptr());
            params.allgather_out = allgather_out.has_value()
                ? reinterpret_cast<void*>(allgather_out.value().data_ptr())
                : nullptr;
            params.max_num_tokens = max_num_tokens;

            int sm_count = get_sm_count();
            int max_sm_to_use_val = max_sm_to_use.has_value()
                ? static_cast<int>(max_sm_to_use.value())
                : sm_count;
            int active_sm_count = std::min(max_sm_to_use_val, sm_count);

            // VEC_SIZE == 8
            constexpr static int vec_size = 8;

            auto clear_kernel = clean_previous_buffer<c_type, NRanks, vec_size>;

            // PDL Launch
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attr[0].val.programmaticStreamSerializationAllowed = 1;
            cudaLaunchConfig_t cfg_1;
            memset(&cfg_1, 0, sizeof(cfg_1));
            cfg_1.gridDim = 32;
            cfg_1.blockDim = 1024;
            cfg_1.dynamicSmemBytes = 0;
            cfg_1.stream = get_stream(allgather_in.device());
            cfg_1.attrs = attr;
            cfg_1.numAttrs = 1;
            cudaError_t err_1 = cudaLaunchKernelEx(&cfg_1, clear_kernel, params);
            TVM_FFI_ICHECK(err_1 == cudaSuccess) << "Failed to launch clear kernel: " << cudaGetErrorString(err_1);


            int cluster_size = 8;
            int block_size = min(max(params.hidden_dim_per_rank / cluster_size / vec_size, 1), 1024);
            int cluster_num = static_cast<int>(token_num);
            FLASHINFER_CHECK(block_size <= 1024 && cluster_size > 0, "Invalid block or cluster size");

            int grid_size = (std::min(active_sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;
            if (grid_size == 0) grid_size = cluster_size;

            cudaLaunchConfig_t cfg;
            memset(&cfg, 0, sizeof(cfg));
            cudaLaunchAttribute attribute[2];
            cfg.gridDim = grid_size;
            cfg.blockDim = block_size;
            cfg.dynamicSmemBytes = 0;
            cfg.stream = get_stream(allgather_in.device());

            int attr_count = 0;
            attribute[attr_count].id = cudaLaunchAttributeProgrammaticStreamSerialization;
            attribute[attr_count].val.programmaticStreamSerializationAllowed = launch_with_pdl ? 1 : 0;
            attr_count++;
            int dev_id;
            cudaGetDevice(&dev_id);
            int major;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev_id);

            if (major >= 90) {
                attribute[attr_count].id = cudaLaunchAttributeClusterDimension;
                attribute[attr_count].val.clusterDim.x = cluster_size;
                attribute[attr_count].val.clusterDim.y = 1;
                attribute[attr_count].val.clusterDim.z = 1;
                attr_count++;
            }
            cfg.attrs = attribute;
            cfg.numAttrs = attr_count;

            TVM_FFI_ICHECK(params.hidden_dim_per_rank % vec_size == 0);
            TVM_FFI_ICHECK(NRanks <= 8);
            TVM_FFI_ICHECK(allgather_in.dtype() == dl_bfloat16 || allgather_in.dtype() == dl_float16);
            TVM_FFI_ICHECK(allgather_out.value().dtype() == dl_bfloat16 || allgather_out.value().dtype() == dl_float16);
            auto kernel = simple_allgather_hidden<c_type, NRanks, false>;
            cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, params);
            TVM_FFI_ICHECK(err == cudaSuccess) << "Failed to launch kernel: " << cudaGetErrorString(err);
            });
        });
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(simple_all_gather, simple_all_gather);
