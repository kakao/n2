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

#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <vector>

#include "spdlog/spdlog.h"

#include "base.h"
#include "mmap.h"
#include "distance.h"
#include "sort.h"
#include "heuristic.h"

namespace n2 {

class VisitedList {
public:
    VisitedList(int size)
    : size_(size), mark_(1) {
        visited_ = new unsigned int[size_];
        memset(visited_, 0, sizeof(unsigned int)*size_);
    }

    inline unsigned int GetVisitMark() const { return mark_; }
    inline unsigned int* GetVisited() const { return visited_; }
    void Reset() {
        if (++mark_ == 0) {
            mark_ = 1;
            memset(visited_, 0, sizeof(unsigned int)*size_);
        }
    }
    ~VisitedList() {
        delete [] visited_;
    }

public:
    unsigned int* visited_;
    unsigned int size_;
    unsigned int mark_;
};

class Hnsw{
public:
    Hnsw();
    Hnsw(int dim, std::string metric = "angular");
    Hnsw(const Hnsw& other);
    Hnsw(Hnsw& other);
    Hnsw(Hnsw&& other) noexcept;
    ~Hnsw();

    Hnsw& operator=(const Hnsw& other);
    Hnsw& operator=(Hnsw&& other) noexcept;
    void SetConfigs(const std::vector<std::pair<std::string, std::string> >& configs);

    bool SaveModel(const std::string& fname) const;
    bool LoadModel(const std::string& fname, const bool use_mmap=true);
    void UnloadModel();

    void AddData(const std::vector<float>& data);

    void Fit();
    void Build(int M = -1, int M0 = -1, int ef_construction = -1, int n_threads = -1, float mult = -1, NeighborSelectingPolicy neighbor_selecting = NeighborSelectingPolicy::HEURISTIC, GraphPostProcessing graph_merging = GraphPostProcessing::SKIP, bool ensure_k = false);

    void SearchByVector(const std::vector<float>& qvec, size_t k, size_t ef_search,
                        std::vector<std::pair<int, float> >& result);

    void SearchById(int id, size_t k, size_t ef_search,
                    std::vector<std::pair<int, float> >& result);

    void PrintDegreeDist() const;
    void PrintConfigs() const;

private:
    int DrawLevel(bool use_default_rng=false);

    void BuildGraph(bool reverse);
    void Insert(HnswNode* qnode);
    void Link(HnswNode* source, HnswNode* target, int level, bool is_naive, size_t dim);
    void SearchAtLayer(const std::vector<float>& qvec, HnswNode* enterpoint,
                       int level, size_t ef,
                       std::priority_queue<FurtherFirst>& result);

    void SearchById_(int cur_node_id, float cur_dist, const float* query_vec,
                     size_t k, size_t ef_search,
                     std::vector<std::pair<int, float> >& result);

    bool SetValuesFromModel(char* model);
    void NormalizeVector(std::vector<float>& vec);
    void MergeEdgesOfTwoGraphs(const std::vector<HnswNode*>& another_nodes);
    size_t GetModelConfigSize() const;
    void SaveModelConfig(char* model);
    template <typename T>
    char* SetValueAndIncPtr(char* ptr, const T& val) {
        *((T*)(ptr)) = val;
        return ptr + sizeof(T);
    }
    template <typename T>
    char* GetValueAndIncPtr(char* ptr, T& val) {
        val = *((T*)(ptr));
        return ptr + sizeof(T);
    }

private:
    std::shared_ptr<spdlog::logger> logger_;
    std::unique_ptr<VisitedList> search_list_;

    const std::string n2_signature = "TOROS_N2@N9R4";
    size_t M_ = 12;
    size_t MaxM_ = 12;
    size_t MaxM0_ = 24;
    size_t efConstruction_ = 150;
    float levelmult_ = 1 / log(1.0*M_);
    int num_threads_ = 1;
    bool ensure_k_ = false;
    bool is_naive_ = false;
    GraphPostProcessing post_ = GraphPostProcessing::SKIP;

    BaseDistance* dist_cls_ = nullptr;
    BaseNeighborSelectingPolicies* selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
    BaseNeighborSelectingPolicies* post_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
    std::uniform_real_distribution<double> uniform_distribution_{0.0, 1.0};
    std::default_random_engine* default_rng_ = nullptr;
    std::mt19937 rng_;

    int maxlevel_ = 0;
    HnswNode* enterpoint_ = nullptr;
    int enterpoint_id_ = 0;
    std::vector<Data> data_;
    std::vector<HnswNode*> nodes_;
    int num_nodes_ = 0;
    DistanceKind metric_;
    char* model_ = nullptr;
    long long model_byte_size_ = 0;
    char* model_higher_level_ = nullptr;
    char* model_level0_ = nullptr;
    size_t data_dim_ = 0;
    long long memory_per_data_ = 0;
    long long memory_per_link_level0_ = 0;
    long long memory_per_node_level0_ = 0;
    long long memory_per_higher_level_ = 0;
    long long memory_per_node_higher_level_ = 0;
    long long higher_level_offset_ = 0;
    long long level0_offset_ = 0;

    Mmap* model_mmap_ = nullptr;

    mutable std::mutex node_list_guard_;
    mutable std::mutex max_level_guard_;

    // configurations
    int rng_seed_ = 17;
    bool use_default_rng_ = false;
};

} // namespace n2
