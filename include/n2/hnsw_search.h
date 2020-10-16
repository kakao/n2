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
#include "hnsw_model.h"

namespace n2 {

class HnswSearch {
public:
    static std::unique_ptr<HnswSearch> GenerateSearcher(std::shared_ptr<const HnswModel> model, size_t data_dim,
                                                        DistanceKind metric);
    virtual ~HnswSearch() {}

    virtual void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                                std::vector<int>& result) = 0;
    virtual void SearchByVector(const std::vector<float>& qvec, size_t k, int ef_search, bool ensure_k,
                                std::vector<std::pair<int, float>>& result) = 0;
    virtual void SearchById(int id, size_t k, int ef_search, bool ensure_k, 
                            std::vector<int>& result) = 0;
    virtual void SearchById(int id, size_t k, int ef_search, bool ensure_k, 
                            std::vector<std::pair<int, float>>& result) = 0;
};

} // namespace n2
