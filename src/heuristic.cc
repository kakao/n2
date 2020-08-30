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

#include "n2/heuristic.h"

#include <xmmintrin.h>

#include <vector>

#include "n2/min_heap.h"

namespace n2 {

using std::priority_queue;
using std::vector;

BaseNeighborSelectingPolicies::~BaseNeighborSelectingPolicies() {}

void NaiveNeighborSelectingPolicies::Select(size_t m, size_t dim, bool select_nn, 
                                            priority_queue<FurtherFirst>& result) {
    while (result.size() > m) {
        result.pop();
    }
}

template<typename DistFuncType>
void HeuristicNeighborSelectingPolicies<DistFuncType>::Select(size_t m, size_t dim, bool select_nn, 
                                                              priority_queue<FurtherFirst>& result) {
    if (result.size() <= m) return;
  
    size_t nn_num = 0;  // # of nearest neighbors
    if (select_nn) {
        nn_num = (size_t)(m * 0.25);
        // m - nn_num neighbors will be chosen as usual with the heuristic algorithm
    }
    size_t nn_picked_num = 0;  // # of nearest neighbors also picked by the heuristic algorithm
    // nn_num - nn_picked_num = # of nearest neighbors selected unconditionally but not picked by the heuristic algorithm 

    vector<FurtherFirst> neighbors, picked;
    MinHeap<float, HnswNode*> skipped;
    while (!result.empty()) {
        neighbors.push_back(result.top());
        result.pop();
    }

    for (auto it = neighbors.rbegin(); it != neighbors.rend(); it++) {
        float cur_dist = it->GetDistance();
        HnswNode* cur_node = it->GetNode();
        _mm_prefetch(cur_node->GetData(), _MM_HINT_T0);
        bool nn_selected = false;
        if (result.size() < nn_num) {
            result.emplace(*it);
            nn_selected = true;
        }

        bool skip = false;
        for (size_t j = 0; j < picked.size(); ++j) {
            if (j < picked.size() - 1) {
                _mm_prefetch(picked[j+1].GetNode()->GetData(), _MM_HINT_T0);
            }
            _mm_prefetch(cur_node->GetData(), _MM_HINT_T1);
            if (dist_func_(cur_node, picked[j].GetNode(), dim) < cur_dist) {
                skip = true;
                break;
            }
        }

        if (!skip) {
            picked.push_back(*it);
            if (nn_selected) {
                // nearest neighbors included in result & picked by the heuristic algorithm
                ++nn_picked_num;
            }
        } else if (!nn_selected && save_remains_) {
            skipped.push(cur_dist, cur_node);
        }
            
        if (picked.size() - nn_picked_num == m - nn_num)
            // check if # of neighbors exclusively picked by the heuristic algorithm equals m - nn_num
            break;
    }

    for (size_t i = nn_picked_num; i < picked.size(); ++i) {
        result.emplace(picked[i]);
    }

    if (save_remains_) {
        while (result.size() < m && skipped.size()) {
            result.emplace(skipped.top().data, skipped.top().key);
            skipped.pop();
        }
    }   
}

template class HeuristicNeighborSelectingPolicies<AngularDistance>;
template class HeuristicNeighborSelectingPolicies<L2Distance>;
template class HeuristicNeighborSelectingPolicies<DotDistance>;

} // namespace n2
