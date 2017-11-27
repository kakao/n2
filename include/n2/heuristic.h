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

#include "common.h"
#include "distance.h"
#include "hnsw_node.h"
#include "sort.h"
#include "min_heap.h"

namespace n2 {

class BaseNeighborSelectingPolicies {
public:
    BaseNeighborSelectingPolicies() {}
    virtual ~BaseNeighborSelectingPolicies() = 0;
    virtual void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) = 0;
};

class NaiveNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    NaiveNeighborSelectingPolicies() {}
    ~NaiveNeighborSelectingPolicies() override {}
    void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) override;
};

class HeuristicNeighborSelectingPolicies : public BaseNeighborSelectingPolicies {
public:
    HeuristicNeighborSelectingPolicies(): save_remains_(false) {}
    HeuristicNeighborSelectingPolicies(bool save_remain) : save_remains_(save_remain) {}
    ~HeuristicNeighborSelectingPolicies() override {}
     void Select(const size_t m, std::priority_queue<FurtherFirst>& result, size_t dim, const BaseDistance* dist_cls) override;
private:
    bool save_remains_;
};

} // namespace n2
