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

#include <memory>
#include <vector>

#include "common.h"
#include "distance.h"
#include "hnsw_model.h"
#include "visited_list.h"

namespace n2 {

class HnswSearch {
public:
    static std::unique_ptr<HnswSearch> GenerateSearcher(std::shared_ptr<const HnswModel> model, size_t data_dim,
                                                        DistanceKind metric);
    virtual ~HnswSearch() {}

    virtual void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                                std::vector<std::pair<int, float>>& result) = 0;
    virtual void SearchById(int id, size_t k, int ef_search, bool ensure_k, 
                            std::vector<std::pair<int, float>>& result) = 0;
};

template<typename DistFuncType>
class HnswSearchImpl : public HnswSearch {
public:
    HnswSearchImpl(std::shared_ptr<const HnswModel> model, size_t data_dim, DistanceKind metric);

    void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                        std::vector<std::pair<int, float>>& result) override;
    void SearchById(int id, size_t k, int ef_search, bool ensure_k,
                    std::vector<std::pair<int, float>>& result) override;


protected:
    void SearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                     bool ensure_k, std::vector<std::pair<int, float>>& result);

protected:
    std::shared_ptr<const HnswModel> model_;
    std::unique_ptr<VisitedList> visited_list_;

    size_t data_dim_;
    DistanceKind metric_;

    DistFuncType dist_func_;
    
    // preallocated buffer
    std::vector<float> normalized_vec_;
    std::vector<std::pair<int, float>> ensure_k_path_;


    // raw pointer of model
    char* model_higher_level_ = nullptr;
    char* model_level0_ = nullptr;
    char* model_level0_node_base_offset_ = nullptr;
    uint64_t memory_per_node_level0_;
    uint64_t memory_per_node_higher_level_;
};

using HnswSearchAngular = HnswSearchImpl<AngularDistance>;
using HnswSearchL2 = HnswSearchImpl<L2Distance>;

} // namespace n2
