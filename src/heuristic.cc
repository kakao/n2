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

#include <xmmintrin.h>

#include "n2/base.h"
#include "n2/heuristic.h"

namespace n2 {

BaseNeighborSelectingPolicies::~BaseNeighborSelectingPolicies() {}

void NaiveNeighborSelectingPolicies::Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) {
    while (result.size() > m) {
        result.pop();
    }
}

void HeuristicNeighborSelectingPolicies::Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) {
    if (result.size() < m) return;
    
    float PORTABLE_ALIGN32 TmpRes[8];
    std::vector<FurtherFirst> neighbors, picked;
    MinHeap<float, HnswNode*> skipped;
    while(!result.empty()) {
        neighbors.push_back(result.top());
        result.pop();
    }

    for (size_t i = 0; i < neighbors.size(); ++i) {
        _mm_prefetch((char*)&(neighbors[i].GetNode()->GetData()), _MM_HINT_T0);
    }

    for (int i = (int)neighbors.size() - 1; i >= 0; --i) {
        bool skip = false;
        float cur_dist = neighbors[i].GetDistance();
        for (size_t j = 0; j < picked.size(); ++j) {
            if (j < picked.size() - 1) {
                _mm_prefetch((char*)&(picked[j+1].GetNode()->GetData()), _MM_HINT_T0);
            }
            _mm_prefetch(&dist_cls, _MM_HINT_T1);
            if (dist_cls->Evaluate((float*)&neighbors[i].GetNode()->GetData()[0], (float*)&picked[j].GetNode()->GetData()[0], dim, TmpRes) < cur_dist) {
                skip = true;
                break;
            }
        }

        if (!skip) {
            picked.push_back(neighbors[i]);
        } else if (save_remains_) {
            skipped.push(cur_dist, neighbors[i].GetNode());
        }
            
        if (picked.size() == m) break;
    }
        
    for(size_t i = 0; i < picked.size(); ++i) {
        result.emplace(picked[i]);
    }
        
    if (save_remains_) {
        while (result.size() < m && skipped.size()) {
            result.emplace(skipped.top().data, skipped.top().key);
            skipped.pop();
        }
    }    
}

} // namespace n2
