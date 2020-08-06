// Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <deque>
#include <memory>
#include <mutex>

#include "common.h"
#include "hnsw_search.h"

namespace n2 {

class HnswSearcherPool {
public:
    static std::unique_ptr<HnswSearcherPool> GeneratePool(std::shared_ptr<const HnswModel> model, size_t data_dim,
                                                          DistanceKind metric) {
        return std::unique_ptr<HnswSearcherPool>(new HnswSearcherPool(std::move(model), data_dim, metric));
    }

    void Clear() {
        pool_.clear();
    }

    std::shared_ptr<HnswSearch> GetInstanceFromPool() {
        {
            std::unique_lock<std::mutex> lock(pool_guard_);
            if (not pool_.empty()) {
                auto s = pool_.back();
                pool_.pop_back();
                return s;
            }
        }
        return HnswSearch::GenerateSearcher(model_, data_dim_, metric_);
    }

    void ReturnInstanceToPool(std::shared_ptr<HnswSearch> s) {
        std::unique_lock<std::mutex> lock(pool_guard_);
        pool_.push_back(std::move(s));
    }

private:
    HnswSearcherPool(std::shared_ptr<const HnswModel> model, size_t data_dim, DistanceKind metric)
        : model_(std::move(model)), data_dim_(data_dim), metric_(metric) {};

private:
    std::shared_ptr<const HnswModel> model_;
    size_t data_dim_;
    DistanceKind metric_;

    std::deque<std::shared_ptr<HnswSearch>> pool_;
    std::mutex pool_guard_;
};

} // namespace n2
