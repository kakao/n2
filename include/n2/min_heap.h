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
#include <algorithm>
#include <stdexcept>

namespace n2 {

template <typename KeyType, typename DataType>
class MinHeap {
public:
    class Item {
    public:
        KeyType key;
        DataType data;
        Item() {}
        Item(const KeyType& key) :key(key) {}
        Item(const KeyType& key, const DataType& data) :key(key), data(data) {}
        bool operator<(const Item& i2) const {
            return key > i2.key;
        }
    };
    
    MinHeap() {
    }
    
    const KeyType top_key() {
        if (v_.size() <= 0) return 0.0;
        return v_[0].key;
    }
    
    Item top() {
        if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
        return v_[0];
    }

    void pop() {
        std::pop_heap(v_.begin(), v_.end());
        v_.pop_back();
    }
    
    void push(const KeyType& key, const DataType& data) {
        v_.emplace_back(Item(key, data));
        std::push_heap(v_.begin(), v_.end());
    }

    size_t size() {
        return v_.size();
    }

protected:
    std::vector<Item> v_;
};

} // namespace n2
