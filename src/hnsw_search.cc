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

#include "n2/hnsw_search.h"

#include <xmmintrin.h>

#include "n2/max_heap.h"
#include "n2/min_heap.h"
#include "n2/utils.h"

namespace n2 {

using std::make_unique;
using std::pair;
using std::runtime_error;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

unique_ptr<HnswSearch> HnswSearch::GenerateSearcher(shared_ptr<const HnswModel> model, size_t data_dim,
                                                    DistanceKind metric) {
    if (metric == DistanceKind::ANGULAR) {
        return make_unique<HnswSearchAngular>(model, data_dim, metric);
    } else if (metric == DistanceKind::L2) {
        return make_unique<HnswSearchL2>(model, data_dim, metric);
    } else {
        throw runtime_error("[Error] Invalid configuration value for DistanceMethod");
    }
}

template<typename DistFuncType>
HnswSearchImpl<DistFuncType>::HnswSearchImpl(shared_ptr<const HnswModel> model, size_t data_dim, DistanceKind metric)
        : model_(model), data_dim_(data_dim), metric_(metric), normalized_vec_(data_dim) {
    visited_list_ = make_unique<VisitedList>(model->GetNumNodes());

    model_higher_level_ = model_->model_higher_level_;
    model_level0_ = model_->model_level0_;
    model_level0_node_base_offset_ = model_->model_level0_node_base_offset_;
    memory_per_node_level0_ = model_->memory_per_node_level0_;
    memory_per_node_higher_level_ = model_->memory_per_node_higher_level_;
}

template<typename DistFuncType>
void HnswSearchImpl<DistFuncType>::SearchByVector(const vector<float>& qvec, size_t k, int ef_search, 
                                                  bool ensure_k, vector<pair<int, float>>& result) {
    if (ef_search < 0)
        ef_search = 50 * k;

    const float* qraw = nullptr;
    if (metric_ == DistanceKind::ANGULAR) {
        Utils::NormalizeVector(qvec, normalized_vec_);
        qraw = &normalized_vec_[0];
    } else {
        qraw = &qvec[0];
    }

    _mm_prefetch(qraw, _MM_HINT_T0);
    int cur_node_id = model_->GetEnterpointId();
    const float* vec = (const float*)(model_level0_node_base_offset_ + cur_node_id * memory_per_node_level0_);
    _mm_prefetch(vec, _MM_HINT_NTA);
    float cur_dist = dist_func_(qraw, vec, data_dim_);
            
    if (ensure_k) {
        ensure_k_path_.clear();
        ensure_k_path_.emplace_back(cur_node_id, cur_dist);
    }

    bool changed;
    for (auto i = model_->GetMaxLevel(); i > 0; --i) {
        visited_list_->Reset();
        unsigned int visited_mark = visited_list_->GetVisitMark();
        unsigned int* visited = visited_list_->GetVisited();
        visited[cur_node_id] = visited_mark;
        
        changed = true;
        while (changed) {
            changed = false;
            int offset = *((int*)(model_level0_ + cur_node_id * memory_per_node_level0_));
            const int* friends_with_size = (const int*)(model_higher_level_ 
                                           + (offset+i-1) * memory_per_node_higher_level_);
            _mm_prefetch(friends_with_size, _MM_HINT_T0);
            int size = friends_with_size[0];
           
            for (auto j = 1; j <= size; ++j) {
                _mm_prefetch(visited + friends_with_size[j], _MM_HINT_T0);
            }
            for (auto j = 1; j <= size; ++j) {
                int fid = friends_with_size[j];
                if (visited[fid] != visited_mark) {
                    _mm_prefetch(qraw, _MM_HINT_T0);
                    const float* vec = (const float*)(model_level0_node_base_offset_ 
                                       + fid * memory_per_node_level0_);
                    _mm_prefetch(vec, _MM_HINT_NTA);
                    visited[fid] = visited_mark;
                    float d = dist_func_(qraw, vec, data_dim_);
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_node_id = fid;
                        changed = true;
                        if (ensure_k) ensure_k_path_.emplace_back(cur_node_id, cur_dist);
                     }
                }
            }
        }
    }

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

template<typename DistFuncType>
void HnswSearchImpl<DistFuncType>::SearchById(int id, size_t k, int ef_search, bool ensure_k, 
                                              vector<pair<int, float>>& result) {
    if (ef_search < 0) {
        ef_search = 50 * k;
    }
    SearchById_(id, 0.0, (const float*)(model_level0_node_base_offset_ + id * memory_per_node_level0_), 
                k, ef_search, ensure_k, result);
}

template<typename DistFuncType>
void HnswSearchImpl<DistFuncType>::SearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, 
                                               size_t ef_search, bool ensure_k, vector<pair<int, float>>& result) {
    IdDistancePairMinHeap candidates;
    IdDistancePairMinHeap visited_nodes;

    candidates.emplace(cur_node_id, cur_dist);

    visited_list_->Reset();
    unsigned int visited_mark = visited_list_->GetVisitMark();
    unsigned int* visited = visited_list_->GetVisited();

    size_t already_visited_for_ensure_k = 0;
    if (ensure_k && !result.empty()) {
        already_visited_for_ensure_k = result.size();
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i].first == cur_node_id) {
                return ;
            }
            visited[result[i].first] = visited_mark;
            visited_nodes.emplace(std::move(result[i]));
        }
        result.clear();
    }
    visited[cur_node_id] = visited_mark;

    float farthest_distance = cur_dist;
    size_t total_size = 1;
    while (!candidates.empty() && visited_nodes.size() < ef_search+already_visited_for_ensure_k) {
        const IdDistancePair& c = candidates.top();
        cur_node_id = c.first;
        visited_nodes.emplace(std::move(const_cast<IdDistancePair&>(c)));
        candidates.pop();

        float minimum_distance = farthest_distance;
        const int* friends_with_size = (const int*)(model_level0_ 
                                        + cur_node_id * memory_per_node_level0_ + sizeof(int));
        _mm_prefetch(friends_with_size, _MM_HINT_T0);
        int size = friends_with_size[0];

        for (auto j = 1; j <= size; ++j) {
            _mm_prefetch(visited + friends_with_size[j], _MM_HINT_T0);
        }
        for (auto j = 1; j <= size; ++j) {
            int node_id = friends_with_size[j];
            if (visited[node_id] != visited_mark) {
                _mm_prefetch(qraw, _MM_HINT_T0);
                const float* vec = (const float*)(model_level0_node_base_offset_ 
                                   + node_id * memory_per_node_level0_);
                _mm_prefetch(vec, _MM_HINT_NTA);
                visited[node_id] = visited_mark;
                float d = dist_func_(qraw, vec, data_dim_);
                if (d < minimum_distance || total_size < ef_search) {
                    candidates.emplace(node_id, d);
                    if (d > farthest_distance) {
                        farthest_distance = d;
                    }
                    ++total_size;
                }
            }
        }
    }

    while (result.size() < k) {
        if (!candidates.empty() && !visited_nodes.empty()) {
            const IdDistancePair& c = candidates.top();
            const IdDistancePair& v = visited_nodes.top();
            if (c.second < v.second) {
                result.emplace_back(std::move(const_cast<IdDistancePair&>(c)));
                candidates.pop();
            } else {
                result.emplace_back(std::move(const_cast<IdDistancePair&>(v)));
                visited_nodes.pop();
            }
        } else if (!candidates.empty()) {
            const IdDistancePair& c = candidates.top();
            result.emplace_back(std::move(const_cast<IdDistancePair&>(c)));
            candidates.pop();
        } else if (!visited_nodes.empty()) {
            const IdDistancePair& v = visited_nodes.top();
            result.emplace_back(std::move(const_cast<IdDistancePair&>(v)));
            visited_nodes.pop();
        } else {
            break;
        }
    }
}

template class HnswSearchImpl<AngularDistance>;
template class HnswSearchImpl<L2Distance>;

} // namespace n2
