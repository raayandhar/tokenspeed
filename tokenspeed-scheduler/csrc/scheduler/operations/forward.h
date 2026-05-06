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

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tokenspeed {

struct ForwardOperationBase {
    std::string request_id;
    std::int32_t request_pool_index;
    std::int32_t input_length;
    // All pages currently occupied by this request (existing + newly allocated).
    std::vector<int32_t> occupied_pages;
    // Index into occupied_pages where newly allocated pages begin.
    std::int32_t begin;
    // Number of newly allocated pages (starting at occupied_pages[begin]).
    std::int32_t size;

    std::int32_t prefill_length;

    // Mamba extension (default: inactive)
    std::int32_t mamba_working_idx{-1};
    std::int32_t mamba_checkpoint_dst_idx{-1};
    std::int32_t mamba_cow_src_idx{-1};
    std::int32_t mamba_branching_seqlen{-1};
};

struct PrefillOperation : public ForwardOperationBase {
    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::int32_t extend_prefix_len;
};

struct DecodeOperation : public ForwardOperationBase {
    std::int32_t decode_input_id = -1;
    // For retraction recover
    std::int32_t hist_token_len = -1;
};

using ForwardOperation = std::variant<PrefillOperation, DecodeOperation>;

struct FlatForwardOperation {
    std::vector<std::string> request_ids;
    std::vector<std::int32_t> request_pool_indices;
    std::vector<std::int32_t> input_lengths;
    // Per-request total number of prompt tokens (Request::PrefillSize()).
    std::vector<std::int32_t> prefill_lengths;

    std::vector<std::vector<std::int32_t>> occupied_pages;
    std::vector<std::int32_t> begins;
    std::vector<std::int32_t> sizes;

    std::vector<std::int32_t> input_ids;
    std::vector<std::int32_t> shifted_input_ids;
    std::vector<std::int32_t> extend_prefix_lens;
    std::vector<std::int32_t> decode_input_ids;
    std::vector<std::int32_t> hist_token_lens;

    // Mamba extension (SoA)
    std::vector<std::int32_t> mamba_working_indices;
    std::vector<std::int32_t> mamba_checkpoint_dst_indices;
    std::vector<std::int32_t> mamba_cow_src_indices;
    std::vector<std::int32_t> mamba_branching_seqlens;

    explicit FlatForwardOperation(std::vector<ForwardOperation> ops) {
        std::stable_partition(ops.begin(), ops.end(),
                              [](const ForwardOperation& a) { return std::holds_alternative<PrefillOperation>(a); });
        for (auto& op : ops) {
            std::visit(
                [this](auto& inner) {
                    request_ids.push_back(std::move(inner.request_id));
                    request_pool_indices.push_back(inner.request_pool_index);
                    input_lengths.push_back(inner.input_length);
                    prefill_lengths.push_back(inner.prefill_length);
                    occupied_pages.push_back(std::move(inner.occupied_pages));
                    begins.push_back(inner.begin);
                    sizes.push_back(inner.size);
                    mamba_working_indices.push_back(inner.mamba_working_idx);
                    mamba_checkpoint_dst_indices.push_back(inner.mamba_checkpoint_dst_idx);
                    mamba_cow_src_indices.push_back(inner.mamba_cow_src_idx);
                    mamba_branching_seqlens.push_back(inner.mamba_branching_seqlen);
                },
                op);
            if (auto* prefill = std::get_if<PrefillOperation>(&op)) {
                input_ids.insert(input_ids.end(), prefill->input_ids.begin(), prefill->input_ids.end());
                shifted_input_ids.insert(shifted_input_ids.end(), prefill->shifted_input_ids.begin(),
                                         prefill->shifted_input_ids.end());
                extend_prefix_lens.push_back(prefill->extend_prefix_len);
            } else if (auto* decode = std::get_if<DecodeOperation>(&op)) {
                decode_input_ids.push_back(decode->decode_input_id);
                hist_token_lens.push_back(decode->hist_token_len);
            }
        }
    }

    bool empty() const { return request_ids.empty(); }
    std::size_t num_extends() const { return extend_prefix_lens.size(); }
};

}  // namespace tokenspeed
