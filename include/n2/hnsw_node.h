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

#include <vector>
#include <mutex>

#include "base.h"

namespace n2 {

class HnswNode {
public:
    explicit HnswNode(int id, const Data* data, int level, int maxsize, int maxsize0);
    void CopyHigherLevelLinksToOptIndex(char* mem_offset, long long memory_per_node_higher_level) const;
    void CopyDataAndLevel0LinksToOptIndex(char* mem_offset, int higher_level_offset, int M0) const;

    inline int GetId() const { return id_; }
    inline int GetLevel() const { return level_; }
    inline const std::vector<float>& GetData() const { return data_->GetData(); }
    inline const std::vector<HnswNode*>& GetFriends(int level) const { return friends_at_layer_[level]; }
    inline void SetFriends(int level, std::vector<HnswNode*>& new_friends) {
            friends_at_layer_[level].swap(new_friends);
    }

private:
    void CopyLinksToOptIndex(char* mem_offset, int level) const;

public:
    int id_;
    const Data* data_;
    int level_;
    size_t maxsize_;
    size_t maxsize0_;
    std::vector<std::vector<HnswNode*>> friends_at_layer_;
    
    std::mutex access_guard_;
};

} // namespace n2
