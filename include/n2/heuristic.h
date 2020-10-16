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

#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>

#include "distance.h"
#include "sort.h"

namespace n2 {

class BaseNeighborSelectingPolicies {
public:
    BaseNeighborSelectingPolicies() {}
    virtual ~BaseNeighborSelectingPolicies() = 0;
    
    virtual void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) = 0;
};

class NaiveNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    NaiveNeighborSelectingPolicies() {}
    ~NaiveNeighborSelectingPolicies() override {}
    void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) override;
};

template<typename DistFuncType>
class HeuristicNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    HeuristicNeighborSelectingPolicies(): save_remains_(false) {}
    HeuristicNeighborSelectingPolicies(bool save_remain) : save_remains_(save_remain) {}
    ~HeuristicNeighborSelectingPolicies() override {}
    /**
     * Returns selected neighbors to result
     * (analagous to SELECT-NEIGHBORS-HEURISTIC in Yu. A. Malkov's paper.)
     *
     * select_nn: if true, select 0.25 * m nearest neighbors to result without applying the heuristic algorithm
     */
    void Select(size_t m, size_t dim, bool select_nn, std::priority_queue<FurtherFirst>& result) override;
private:
    bool save_remains_;
    DistFuncType dist_func_;
};

} // namespace n2
