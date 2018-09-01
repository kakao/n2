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
#include <string.h>

namespace n2 {

class MinHeap2
{
public:
    class Item {
    public:
        float key;
        int data;
        Item() {}
        Item(const float& key) :key(key) {}
        Item(const float& key, const int& data) :key(key), data(data) {}
        bool operator < (const Item& that) const {
            return this->key < that.key;
        }
        bool operator < (const float& that) const {
            return this->key < that;
        }
    };

    int K;
    int end;
    float max_val;
    Item* nns;

    MinHeap2(int _K) {
        alloc(_K);
        end = 0;
        max_val = 987654321.0f;
    }

    ~MinHeap2() {
        free();
    }

    void alloc(int _K){
        K = _K;
        nns = new Item[K];
    }

    void free() {
        if (K) {
            delete[] nns;
        }
    }

    Item top() {
        if (end <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
        return nns[0];
    }

    void pop() {
        if (end > 0) {
            end -= 1;
            memcpy((void*)&nns[0], (void*)&nns[1], sizeof(Item) * end);
        }
    }

    void push(const float& key, const int& val) {
        if (end >= K && key >= max_val)
            return;
        Item* ptr = std::lower_bound(nns, nns + end, key);
        int idx = (int)(ptr - nns);
        if (idx >= K)
            return;
        if (end < idx + 1)
            end = idx + 1;
        max_val = std::max(max_val, key);
        if (idx + 1 == K) {
            nns[idx].key = key;
            nns[idx].data = val;
        }
        else {
            memmove((void*)&nns[idx + 1], (void*)&nns[idx], sizeof(Item) * (K - idx - 1));
            nns[idx].key = key;
            nns[idx].data = val;
        }
    }

    int size() {
        return end;
    }
};

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

public:
    std::vector<Item> v_;
};

template <typename KeyType, typename DataType>
class MinHeap3 {
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
    
    MinHeap3() {
        heapify_ = true;
    }
    
    const KeyType top_key() {
        if (!heapify_) make_heap();
        if (v_.size() <= 0) return 0.0;
        return v_[0].key;
    }
    
    Item top() {
        if (!heapify_) make_heap();
        if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
        return v_[0];
    }

    void pop() {
        if (!heapify_) make_heap();
        std::pop_heap(v_.begin(), v_.end());
        v_.pop_back();
    }
    
    void push(const KeyType& key, const DataType& data) {
        v_.emplace_back(Item(key, data));
        heapify_ = false;
    }

    void make_heap()
    {
        std::make_heap(v_.begin(), v_.end());
        heapify_ = true;
    }

    size_t size() {
        return v_.size();
    }

protected:
    std::vector<Item> v_;
    bool heapify_;
};



struct lesser
{
    template<class T>
    bool operator()(T const &a, T const &b) const {
        return a.key < b.key;
    }
};

template <typename KeyType, typename DataType>
class MinHeap4 {
public:

    class Item {
    public:
        KeyType key;
        DataType data;
        Item() {}
        Item(const KeyType& key) :key(key) {}
        Item(const KeyType& key, const DataType& data) :key(key), data(data) {}
        bool operator < (const Item& i2) const {
            return key > i2.key;
        }
    };
    
    MinHeap4() {
        sorted_ = true;
    }
    
    Item top() {
        if (!sorted_) sort();
        if (v_.size() <= 0) throw std::runtime_error("[Error] Called top() operation with empty heap");
        return v_.back();
    }

    void pop() {
        if (!sorted_) sort();
        v_.pop_back();
    }
    
    void push(const KeyType& key, const DataType& data) {
        v_.emplace_back(Item(key, data));
        sorted_ = false;
    }

    void sort()
    {
        std::sort(v_.begin(), v_.end(), lesser());
        sorted_ = true;
    }

    size_t size() {
        return v_.size();
    }

protected:
    std::vector<Item> v_;
    bool sorted_;
};


} // namespace n2
