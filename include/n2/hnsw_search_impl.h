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
#include "hnsw_search.h"
#include "min_heap.h"
#include "visited_list.h"

namespace n2 {

template<typename DistFuncType>
class HnswSearchImpl : public HnswSearch {
public:
    HnswSearchImpl(std::shared_ptr<const HnswModel> model, size_t data_dim, DistanceKind metric);

    void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                        std::vector<int>& result) override;
    void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                        std::vector<std::pair<int, float>>& result) override;
    void SearchById(int id, size_t k, int ef_search, bool ensure_k,
                    std::vector<int>& result) override;
    void SearchById(int id, size_t k, int ef_search, bool ensure_k,
                    std::vector<std::pair<int, float>>& result) override;

protected:
    template<typename ResultType>
    void SearchByVector_(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                         ResultType& result);

    inline void CallSearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                                bool ensure_k, std::vector<int>& result) {
        if (ensure_k) {
            std::vector<std::pair<int, float>> tmp_result;
            CallSearchById_(cur_node_id, cur_dist, qraw, k, ef_search, ensure_k, tmp_result);
            for (const auto& p : tmp_result) {
                result.push_back(p.first);
            }
        } else {
            SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, false, result);
        }
    }
    inline void CallSearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                                bool ensure_k, std::vector<std::pair<int, float>>& result) {
        if (ensure_k) {
            while (result.size() < k && !ensure_k_path_.empty()) {
                cur_node_id = ensure_k_path_.back().first;
                cur_dist = ensure_k_path_.back().second;
                ensure_k_path_.pop_back();
                SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, ensure_k, result);
            }
        } else {
            SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, ensure_k, result);
        }
    }

    template<typename ResultType>
    inline void SearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                            bool ensure_k, ResultType& result) {
        if (ef_search < k) 
            SearchByIdV1_(cur_node_id, cur_dist, qraw, k, ef_search, ensure_k, result);
        else
            SearchByIdV2_(cur_node_id, cur_dist, qraw, k, ef_search, ensure_k, result);
    }

    template<typename ResultType>
    void SearchByIdV1_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                       bool ensure_k, ResultType& result);

    template<typename ResultType>
    void SearchByIdV2_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search,
                       bool ensure_k, ResultType& result);

    bool PrepareEnsureKSearch(int cur_node_id, std::vector<int>& result, IdDistancePairMinHeap& visited_nodes);
    bool PrepareEnsureKSearch(int cur_node_id, std::vector<std::pair<int, float>>& result,
                              IdDistancePairMinHeap& visited_nodes);

    void MakeSearchResult(size_t k, IdDistancePairMinHeap& candidates, IdDistancePairMinHeap& visited_nodes,
                          std::vector<int>& result);
    void MakeSearchResult(size_t k, IdDistancePairMinHeap& candidates, IdDistancePairMinHeap& visited_nodes,
                          std::vector<std::pair<int, float>>& result);

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
using HnswSearchDot = HnswSearchImpl<DotDistance>;

} // namespace n2
