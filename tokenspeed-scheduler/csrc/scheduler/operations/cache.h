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
#include <functional>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "resource/types.h"

namespace tokenspeed {

struct CacheOperationBase {
    cache_op_id op_id = 0;
    std::vector<std::int32_t> src_pages;
    std::vector<std::int32_t> dst_pages;
};

struct PrefetchOperation : public CacheOperationBase {
    std::string request_id;
    std::vector<std::string> rolling_page_hashes;
};
struct BackUpOperation : public CacheOperationBase {
    std::vector<std::string> rolling_page_hashes;
};
struct WriteBackOperation {
    cache_op_id op_id{0};
    std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer;  // (src_device, dst_host)
    bool is_retract{false};
};

struct FlatWriteBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;
    std::vector<bool> is_retract;

    explicit FlatWriteBackOperation(const std::vector<WriteBackOperation>& ops) {
        struct PairHash {
            std::size_t operator()(const std::tuple<std::int32_t, std::int32_t>& t) const {
                std::size_t h1 = std::hash<std::int32_t>{}(std::get<0>(t));
                std::size_t h2 = std::hash<std::int32_t>{}(std::get<1>(t));
                return h1 ^ (h2 << 32) ^ (h2 >> 32);
            }
        };
        std::unordered_set<std::tuple<std::int32_t, std::int32_t>, PairHash> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_pages_this_op;
            std::vector<std::int32_t> dst_pages_this_op;
            for (const auto& page : op.pages_to_transfer) {
                if (seen.insert(page).second) {
                    src_pages_this_op.push_back(std::get<0>(page));
                    dst_pages_this_op.push_back(std::get<1>(page));
                }
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_pages_this_op));
            dst_pages.push_back(std::move(dst_pages_this_op));
            is_retract.push_back(op.is_retract);
        }
    }
};

struct LoadBackOperation {
    cache_op_id op_id{0};
    std::vector<std::tuple<std::int32_t, std::int32_t>> pages_to_transfer;
};

struct FlatLoadBackOperation {
    std::vector<cache_op_id> op_ids;
    std::vector<std::vector<std::int32_t>> src_pages;
    std::vector<std::vector<std::int32_t>> dst_pages;

    explicit FlatLoadBackOperation(const std::vector<LoadBackOperation>& ops) {
        struct PairHash {
            std::size_t operator()(const std::tuple<std::int32_t, std::int32_t>& t) const {
                std::size_t h1 = std::hash<std::int32_t>{}(std::get<0>(t));
                std::size_t h2 = std::hash<std::int32_t>{}(std::get<1>(t));
                return h1 ^ (h2 << 32) ^ (h2 >> 32);
            }
        };
        std::unordered_set<std::tuple<std::int32_t, std::int32_t>, PairHash> seen;
        for (const auto& op : ops) {
            std::vector<std::int32_t> src_pages_this_op;
            std::vector<std::int32_t> dst_pages_this_op;
            for (const auto& page : op.pages_to_transfer) {
                if (seen.insert(page).second) {
                    src_pages_this_op.push_back(std::get<0>(page));
                    dst_pages_this_op.push_back(std::get<1>(page));
                }
            }
            op_ids.push_back(op.op_id);
            src_pages.push_back(std::move(src_pages_this_op));
            dst_pages.push_back(std::move(dst_pages_this_op));
        }
    }
};

using CacheOperation = std::variant<PrefetchOperation, FlatLoadBackOperation, BackUpOperation, FlatWriteBackOperation>;

}  // namespace tokenspeed
