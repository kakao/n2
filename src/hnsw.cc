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

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>
#include <xmmintrin.h>

#include "n2/hnsw.h"
#include "n2/hnsw_node.h"
#include "n2/distance.h"
#include "n2/min_heap.h"

#define MERGE_BUFFER_ALGO_SWITCH_THRESHOLD 100

namespace n2 {

using std::endl;
using std::fstream;
using std::max;
using std::min;
using std::mutex;
using std::ofstream;
using std::ifstream;
using std::pair;
using std::priority_queue;
using std::setprecision;
using std::string;
using std::stof;
using std::stoi;
using std::to_string;
using std::unique_lock;
using std::unordered_set;
using std::vector;


thread_local VisitedList* visited_list_ = nullptr;

Hnsw::Hnsw() {
    logger_ = spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    metric_ = DistanceKind::ANGULAR;
    dist_cls_ = new AngularDistance();
}

Hnsw::Hnsw(int dim, string metric) : data_dim_(dim) {
    logger_ = spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    if (metric == "L2" || metric =="euclidean") {
        metric_ = DistanceKind::L2;
        dist_cls_ = new L2Distance();
    } else if (metric == "angular") {
        metric_ = DistanceKind::ANGULAR;
        dist_cls_ = new AngularDistance();
    } else {
        throw std::runtime_error("[Error] Invalid configuration value for DistanceMethod: " + metric);
    }
}

Hnsw::Hnsw(const Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw::Hnsw(Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw::Hnsw(Hnsw&& other) noexcept {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    model_byte_size_ = other.model_byte_size_;
    model_ = other.model_;
    other.model_ = nullptr;
    model_mmap_ = other.model_mmap_;
    other.model_mmap_ = nullptr;
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
}

Hnsw& Hnsw::operator=(const Hnsw& other) {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }

    if(model_) {
        delete [] model_;
        model_ = nullptr;
    }

    if(dist_cls_) {
       delete dist_cls_;
       dist_cls_ = nullptr;
    }

    model_byte_size_ = other.model_byte_size_;
    model_ = new char[model_byte_size_];
    std::copy(other.model_, other.model_ + model_byte_size_, model_);
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
    return *this;
}

Hnsw& Hnsw::operator=(Hnsw&& other) noexcept {
    logger_= spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
    if(model_mmap_) {
        delete model_mmap_;
        model_mmap_ = nullptr;
    } else {
        delete [] model_;
        model_ = nullptr;
    }

    if(dist_cls_) {
       delete dist_cls_;
       dist_cls_ = nullptr;
    }

    model_byte_size_ = other.model_byte_size_;
    model_ = other.model_;
    other.model_ = nullptr;
    model_mmap_ = other.model_mmap_;
    other.model_mmap_ = nullptr;
    SetValuesFromModel(model_);
    search_list_.reset(new VisitedList(num_nodes_));
    if(metric_ == DistanceKind::ANGULAR) {
        dist_cls_ = new AngularDistance();
    } else if (metric_ == DistanceKind::L2) {
        dist_cls_ = new L2Distance();
    }
    return *this;
}

Hnsw::~Hnsw() {
    if (model_mmap_) {
        delete model_mmap_;
    } else {
        if (model_)
            delete[] model_;
    }

    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }

    if (default_rng_) {
        delete default_rng_;
    }

    if (dist_cls_) {
        delete dist_cls_;
    }

    if (selecting_policy_cls_) {
        delete selecting_policy_cls_;
    }

    if (post_policy_cls_) {
        delete post_policy_cls_;
    }
}


void Hnsw::SetConfigs(const vector<pair<string, string> >& configs) {
    bool is_levelmult_set = false;
    for (const auto& c : configs) {
        if (c.first == "M") {
            MaxM_ = M_ = (size_t)stoi(c.second);
        } else if (c.first == "MaxM0") {
            MaxM0_ = (size_t)stoi(c.second);
        } else if (c.first == "efConstruction") {
            efConstruction_ = (size_t)stoi(c.second);
        } else if (c.first == "NumThread") {
            num_threads_ = stoi(c.second);
        } else if (c.first == "Mult") {
            levelmult_ = stof(c.second);
            is_levelmult_set = true;
        } else if (c.first == "NeighborSelecting") {

            if(selecting_policy_cls_) delete selecting_policy_cls_;

            if (c.second == "heuristic") {
                selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
                is_naive_ = false;
            } else if (c.second == "heuristic_save_remains") {
                selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
                is_naive_ = false;
            } else if (c.second == "naive") {
                selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
                is_naive_ = true;
            } else {
                throw std::runtime_error("[Error] Invalid configuration value for NeighborSelecting: " + c.second);
            }
        } else if (c.first == "GraphMerging") {
            if (c.second == "skip") {
                post_ = GraphPostProcessing::SKIP;
            } else if (c.second == "merge_level0") {
                post_ = GraphPostProcessing::MERGE_LEVEL0;
            } else {
                throw std::runtime_error("[Error] Invalid configuration value for GraphMerging: " + c.second);
            }
        } else if (c.first == "EnsureK") {
            if (c.second == "true") {
                ensure_k_ = true;
            } else {
                ensure_k_ = false;
            }
        } else {
            throw std::runtime_error("[Error] Invalid configuration key: " + c.first);
        }
    }
    if (!is_levelmult_set) {
        levelmult_ = 1 / log(1.0*M_);
    }
}

int Hnsw::DrawLevel(bool use_default_rng) {
    double r = use_default_rng ? uniform_distribution_(*default_rng_) : uniform_distribution_(rng_);
    if (r < std::numeric_limits<double>::epsilon())
        r = 1.0;
    return (int)(-log(r) * levelmult_);
}

void Hnsw::Build(int M, int MaxM0, int ef_construction, int n_threads, float mult, NeighborSelectingPolicy neighbor_selecting, GraphPostProcessing graph_merging, bool ensure_k) {
    if ( M > 0 ) MaxM_ = M_ = M;
    if ( MaxM0 > 0 ) MaxM0_ = MaxM0;
    if ( ef_construction > 0 ) efConstruction_ = ef_construction;
    if ( n_threads > 0 ) num_threads_ = n_threads;
    levelmult_ = mult > 0 ? mult : 1 / log(1.0*M_);

    if (selecting_policy_cls_) delete selecting_policy_cls_;
    if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC) {
        selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(false);
        is_naive_ = false;
    } else if (neighbor_selecting == NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS) {
        selecting_policy_cls_ = new HeuristicNeighborSelectingPolicies(true);
        is_naive_ = false;
    } else if (neighbor_selecting == NeighborSelectingPolicy::NAIVE) {
        selecting_policy_cls_ = new NaiveNeighborSelectingPolicies();
        is_naive_ = true;
    }
    post_ = graph_merging;
    ensure_k_ = ensure_k;
    Fit();
}

void Hnsw::Fit() {
    if (data_.size() == 0) throw std::runtime_error("[Error] No data to fit. Load data first.");
    if (default_rng_ == nullptr)
        default_rng_ = new std::default_random_engine(100);
    rng_.seed(rng_seed_);
    BuildGraph(false);
    if (post_ == GraphPostProcessing::MERGE_LEVEL0) {
        vector<HnswNode*> nodes_backup;
        nodes_backup.swap(nodes_);
        BuildGraph(true);
        MergeEdgesOfTwoGraphs(nodes_backup);
        for (size_t i = 0; i < nodes_backup.size(); ++i) {
            delete nodes_backup[i];
        }
        nodes_backup.clear();
    }

    long long totalLevel = 0;
    for(size_t i = 0; i < nodes_.size(); ++i) {
        totalLevel += nodes_[i]->GetLevel();
    }
    enterpoint_id_ = enterpoint_->GetId();
    num_nodes_ = nodes_.size();
    long long model_config_size = GetModelConfigSize();
    memory_per_node_higher_level_ = sizeof(int) * (1 + MaxM_);  // "1" for saving num_links
    long long higher_level_size = memory_per_node_higher_level_ * totalLevel;
    memory_per_data_ = sizeof(float) * data_dim_;
    memory_per_link_level0_ = sizeof(int) * (1 + 1 + MaxM0_);  // "1" for offset pos, 1" for saving num_links
    memory_per_node_level0_ = memory_per_link_level0_ + memory_per_data_;
    long long level0_size = memory_per_node_level0_ * data_.size();

    model_byte_size_ = model_config_size + level0_size + higher_level_size;
    model_ = new char[model_byte_size_];
    if (model_ == NULL) {
        throw std::runtime_error("[Error] Fail to allocate memory for optimised index (size: "
                                 + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");
    }
    memset(model_, 0, model_byte_size_);
    model_level0_ = model_ + model_config_size;
    model_higher_level_ = model_level0_ + level0_size;

    SaveModelConfig(model_);
    int higher_offset = 0;
    for (size_t i = 0; i < nodes_.size(); ++i) {
        int level = nodes_[i]->GetLevel();
        if(level > 0) {
            nodes_[i]->CopyDataAndLevel0LinksToOptIndex(model_level0_ + i * memory_per_node_level0_, higher_offset, MaxM0_);
            nodes_[i]->CopyHigherLevelLinksToOptIndex(model_higher_level_ + memory_per_node_higher_level_*higher_offset, memory_per_node_higher_level_);
            higher_offset += nodes_[i]->GetLevel();
        } else {
            nodes_[i]->CopyDataAndLevel0LinksToOptIndex(model_level0_ + i * memory_per_node_level0_, 0, MaxM0_);
        }

    }
    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }
    nodes_.clear();
    data_.clear();
}

void Hnsw::BuildGraph(bool reverse) {
    nodes_.resize(data_.size());
    int level = DrawLevel(use_default_rng_);
    HnswNode* first = new HnswNode(0, &(data_[0]), level, MaxM_, MaxM0_);
    nodes_[0] = first;
    maxlevel_ = level;
    enterpoint_ = first;
    if (reverse) {
        #pragma omp parallel num_threads(num_threads_)
        {
            visited_list_ = new VisitedList(data_.size());

            #pragma omp for schedule(dynamic,128)
            for (size_t i = data_.size() - 1; i >= 1; --i) {
                level = DrawLevel(use_default_rng_);
                HnswNode* qnode = new HnswNode(i, &data_[i], level, MaxM_, MaxM0_);
                nodes_[i] = qnode;
                Insert(qnode);
            }
            delete visited_list_;
            visited_list_ = nullptr;
        }
    } else {
        #pragma omp parallel num_threads(num_threads_)
        {
            visited_list_ = new VisitedList(data_.size());
            #pragma omp for schedule(dynamic,128)
            for (size_t i = 1; i < data_.size(); ++i) {
                level = DrawLevel(use_default_rng_);
                HnswNode* qnode = new HnswNode(i, &data_[i], level, MaxM_, MaxM0_);
                nodes_[i] = qnode;
                Insert(qnode);
            }
            delete visited_list_;
            visited_list_ = nullptr;
        }
    }

    search_list_.reset(new VisitedList(data_.size()));
}

bool Hnsw::SaveModel(const string& fname) const {
    ofstream b_stream(fname.c_str(), fstream::out|fstream::binary);
    if (b_stream) {
        b_stream.write(model_, model_byte_size_);
        return (b_stream.good());
    } else {
        throw std::runtime_error("[Error] Failed to save model to file: " + fname);
    }
    return false;
}

bool Hnsw::LoadModel(const string& fname, const bool use_mmap) {
    if(!use_mmap) {
        ifstream in;
        in.open(fname, fstream::in|fstream::binary|fstream::ate);
        if(in.is_open()) {
            size_t size = in.tellg();
            in.seekg(0, fstream::beg);
            model_ = new char[size];
            model_byte_size_ = size;
            in.read(model_, size);
            in.close();
        } else {
            throw std::runtime_error("[Error] Failed to load model to file: " + fname+ " not found!");
        }
    } else {
        model_mmap_ = new Mmap(fname.c_str());
        model_byte_size_ = model_mmap_->GetFileSize();
        model_ = model_mmap_->GetData();
    }
    char* ptr = model_;
    ptr = GetValueAndIncPtr<size_t>(ptr, M_);
    ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
    ptr = GetValueAndIncPtr<size_t>(ptr, MaxM0_);
    ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
    ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
    ptr = GetValueAndIncPtr<int>(ptr, maxlevel_);
    ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
    size_t model_data_dim = *((size_t*)(ptr));
    if (data_dim_ > 0 && model_data_dim != data_dim_) {
        throw std::runtime_error("[Error] index dimension(" + to_string(data_dim_)
                                 + ") != model dimension(" + to_string(model_data_dim) + ")");
    }
    ptr = GetValueAndIncPtr<size_t>(ptr, data_dim_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
    ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_higher_level_);
    ptr = GetValueAndIncPtr<long long>(ptr, higher_level_offset_);
    ptr = GetValueAndIncPtr<long long>(ptr, level0_offset_);
    long long level0_size = memory_per_node_level0_ * num_nodes_;
    long long model_config_size = GetModelConfigSize();
    model_level0_ = model_ + model_config_size;
    model_higher_level_ = model_level0_ + level0_size;
    search_list_.reset(new VisitedList(num_nodes_));
    if(dist_cls_) {
        delete dist_cls_;
    }
    switch (metric_) {
        case DistanceKind::ANGULAR:
            dist_cls_ = new AngularDistance();
            break;
        case DistanceKind::L2:
            dist_cls_ = new L2Distance();
            break;
        default:
            throw std::runtime_error("[Error] Unknown distance metric. ");
    }
    return true;
}

void Hnsw::UnloadModel() {
    if (model_mmap_ != nullptr) {
        model_mmap_->UnMap();
        delete model_mmap_;
        model_mmap_ = nullptr;
        model_ = nullptr;
        model_higher_level_ = nullptr;
        model_level0_ = nullptr;
    }

    search_list_.reset(nullptr);

    if (visited_list_ != nullptr) {
        delete visited_list_;
        visited_list_ = nullptr;
    }
}

void Hnsw::AddData(const std::vector<float>& data) {
    if (model_ != nullptr) {
        throw std::runtime_error("[Error] This index already has a trained model. Adding an item is not allowed.");
    }

    if (data.size() != data_dim_) {
        throw std::runtime_error("[Error] Invalid dimension data inserted: " + to_string(data.size()) + ", Predefined dimension: " + to_string(data_dim_));
    }

    if(metric_ == DistanceKind::ANGULAR) {
        vector<float> data_copy(data);
        NormalizeVector(data_copy);
        data_.emplace_back(data_copy);
    } else {
        data_.emplace_back(data);
    }
}

void Hnsw::Insert(HnswNode* qnode) {
    int cur_level = qnode->GetLevel();
    unique_lock<mutex> *lock = nullptr;
    if (cur_level > maxlevel_) lock = new unique_lock<mutex>(max_level_guard_);

    int maxlevel_copy = maxlevel_;
    HnswNode* enterpoint = enterpoint_;
    const std::vector<float>& qvec = qnode->GetData();
    const float* qraw = &qvec[0];
    float PORTABLE_ALIGN32 TmpRes[8];
    if (cur_level < maxlevel_copy) {
        _mm_prefetch(&dist_cls_, _MM_HINT_T0);
        HnswNode* cur_node = enterpoint;
        float d = dist_cls_->Evaluate(qraw, (float*)&cur_node->GetData()[0], data_dim_, TmpRes);
        float cur_dist = d;
        for (int i = maxlevel_copy; i > cur_level; --i) {
            bool changed = true;
            while (changed) {
                changed = false;
                unique_lock<mutex> local_lock(cur_node->access_guard_);
                const vector<HnswNode*>& neighbors = cur_node->GetFriends(i);
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    _mm_prefetch((char*)&((*iter)->GetData()), _MM_HINT_T0);
                }

                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    d = dist_cls_->Evaluate(qraw, &(*iter)->GetData()[0], data_dim_, TmpRes);
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_node = (*iter);
                        changed = true;
                    }
                }
            }
        }
        enterpoint = cur_node;
    }
    _mm_prefetch(&selecting_policy_cls_, _MM_HINT_T0);
    for (int i = std::min(maxlevel_copy, cur_level); i >= 0; --i) {
        priority_queue<FurtherFirst> temp_res;
        SearchAtLayer(qvec, enterpoint, i, efConstruction_, temp_res);
        selecting_policy_cls_->Select(M_, temp_res, data_dim_, dist_cls_);
        while (temp_res.size() > 0) {
            auto* top_node = temp_res.top().GetNode();
            temp_res.pop();
            Link(top_node, qnode, i, is_naive_, data_dim_);
            Link(qnode, top_node, i, is_naive_, data_dim_);
        }
    }
    if (cur_level > enterpoint_->GetLevel()) {
        maxlevel_ = cur_level;
        enterpoint_ = qnode;
    }
    if (lock != nullptr) delete lock;
}

void Hnsw::Link(HnswNode* source, HnswNode* target, int level, bool is_naive, size_t dim) {
    std::unique_lock<std::mutex> lock(source->access_guard_);
    std::vector<HnswNode*>& neighbors = source->friends_at_layer_[level];
    neighbors.push_back(target);
    bool shrink = (level > 0 && neighbors.size() > source->maxsize_) || (level <= 0 && neighbors.size() > source->maxsize0_);
    if (!shrink) return;
    float PORTABLE_ALIGN32 TmpRes[8];
    if (is_naive) {
        float max = dist_cls_->Evaluate((float*)&source->GetData()[0], (float*)&neighbors[0]->GetData()[0], dim, TmpRes);
        int maxi = 0;
        for (size_t i = 1; i < neighbors.size(); ++i) {
                float curd = dist_cls_->Evaluate((float*)&source->GetData()[0], (float*)&neighbors[i]->GetData()[0], dim, TmpRes);
                if (curd > max) {
                    max = curd;
                    maxi = i;
                }
        }
        neighbors.erase(neighbors.begin() + maxi);
    } else {
        std::priority_queue<FurtherFirst> tempres;
        for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
            _mm_prefetch((char*)&((*iter)->GetData()), _MM_HINT_T0);
        }

        for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
            tempres.emplace((*iter), dist_cls_->Evaluate((float*)&source->data_->GetData()[0], (float*)&(*iter)->GetData()[0], dim, TmpRes));
        }
        selecting_policy_cls_->Select(tempres.size() - 1, tempres, dim, dist_cls_);
        neighbors.clear();
        while (tempres.size()) {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
   }
}

void Hnsw::MergeEdgesOfTwoGraphs(const vector<HnswNode*>& another_nodes) {
#pragma omp parallel for schedule(dynamic,128) num_threads(num_threads_)
    for (size_t i = 1; i < data_.size(); ++i) {
        const vector<HnswNode*>& neighbors1 = nodes_[i]->GetFriends(0);
        const vector<HnswNode*>& neighbors2 = another_nodes[i]->GetFriends(0);
        unordered_set<int> merged_neighbor_id_set = unordered_set<int>();
        for (HnswNode* cur : neighbors1) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        for (HnswNode* cur : neighbors2) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        priority_queue<FurtherFirst> temp_res;
        const std::vector<float>& ivec = data_[i].GetData();
        float PORTABLE_ALIGN32 TmpRes[8];
        for (int cur : merged_neighbor_id_set) {
            temp_res.emplace(nodes_[cur], dist_cls_->Evaluate((float*)&data_[cur].GetData()[0], (float*)&ivec[0], data_dim_, TmpRes));
        }

        // Post Heuristic
        post_policy_cls_->Select(MaxM0_, temp_res, data_dim_, dist_cls_);
        vector<HnswNode*> merged_neighbors = vector<HnswNode*>();
        while (!temp_res.empty()) {
            merged_neighbors.emplace_back(temp_res.top().GetNode());
            temp_res.pop();
        }
        nodes_[i]->SetFriends(0, merged_neighbors);
    }
}

void Hnsw::NormalizeVector(std::vector<float>& vec) {
   float sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
   if (sum != 0.0) {
       sum = 1 / sqrt(sum);
       std::transform(vec.begin(), vec.end(), vec.begin(), std::bind1st(std::multiplies<float>(), sum));
   }
}

void Hnsw::SearchById_(int cur_node_id, float cur_dist, const float* qraw, size_t k, size_t ef_search, vector<pair<int, float> >& result) {
    MinHeap<float, int> dh;
    dh.push(cur_dist, cur_node_id);
    float PORTABLE_ALIGN32 TmpRes[8];

    typedef typename MinHeap<float, int>::Item  QueueItem;
    std::queue<QueueItem> q;
    search_list_->Reset();

    unsigned int mark = search_list_->GetVisitMark();
    unsigned int* visited = search_list_->GetVisited();
    bool need_sort = false;
    if (ensure_k_) {
        if (!result.empty()) need_sort = true;
        for (size_t i = 0; i < result.size(); ++i)
            visited[result[i].first] = mark;
        if (visited[cur_node_id] == mark) return;
    }
    visited[cur_node_id] = mark;

    std::priority_queue<pair<float, int> > visited_nodes;

    int tnum;
    float d;
    QueueItem e;
    float maxKey = cur_dist;
    size_t total_size = 1;
    while (dh.size() > 0 && visited_nodes.size() < (ef_search >> 1)) {
        e = dh.top();
        dh.pop();
        cur_node_id = e.data;

        visited_nodes.emplace(e.key, e.data);

        float topKey = maxKey;

        int *data = (int*)(model_level0_ + cur_node_id*memory_per_node_level0_ + sizeof(int));
        int size = *data;
        for (int j = 1; j <= size; ++j) {
            tnum = *(data + j);
            _mm_prefetch(dist_cls_, _MM_HINT_T0);
            if (visited[tnum] != mark) {
                visited[tnum] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)(model_level0_ + tnum*memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
                if (d < topKey || total_size < ef_search) {
                    q.emplace(QueueItem(d, tnum));
                    ++total_size;
                }
            }
        }
        while(!q.empty()) {
            dh.push(q.front().key, q.front().data);
            if (q.front().key > maxKey) maxKey = q.front().key;
            q.pop();
        }
    }

    vector<pair<float, int> > res_t;
    while(dh.size() && res_t.size() < k) {
        res_t.emplace_back(dh.top().key, dh.top().data);
        dh.pop();
    }
    while (visited_nodes.size() > k) visited_nodes.pop();
    while (!visited_nodes.empty()) {
        res_t.emplace_back(visited_nodes.top());
        visited_nodes.pop();
    }
    _mm_prefetch(&res_t[0], _MM_HINT_T0);
    std::sort(res_t.begin(), res_t.end());
    size_t sz;
    if (ensure_k_) {
        sz = min(k - result.size(), res_t.size());
    } else {
        sz = min(k, res_t.size());
    }
    for(size_t i = 0; i < sz; ++i)
        result.push_back(pair<int, float>(res_t[i].second, res_t[i].first));
    if (ensure_k_ && need_sort) {
        _mm_prefetch(&result[0], _MM_HINT_T0);
        sort(result.begin(), result.end(), [](const pair<int, float>& i, const pair<int, float>& j) -> bool {
                return i.second < j.second; });
    }
}

bool Hnsw::SetValuesFromModel(char* model) {
    if(model) {
        char* ptr = model;
        ptr = GetValueAndIncPtr<size_t>(ptr, M_);
        ptr = GetValueAndIncPtr<size_t>(ptr, MaxM_);
        ptr = GetValueAndIncPtr<size_t>(ptr, MaxM0_);
        ptr = GetValueAndIncPtr<size_t>(ptr, efConstruction_);
        ptr = GetValueAndIncPtr<float>(ptr, levelmult_);
        ptr = GetValueAndIncPtr<int>(ptr, maxlevel_);
        ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
        ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
        ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
        ptr += sizeof(size_t);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_data_);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
        ptr = GetValueAndIncPtr<long long>(ptr, memory_per_node_higher_level_);
        ptr = GetValueAndIncPtr<long long>(ptr, higher_level_offset_);
        ptr = GetValueAndIncPtr<long long>(ptr, level0_offset_);
        long long level0_size = memory_per_node_level0_ * num_nodes_;
        long long model_config_size = GetModelConfigSize();
        model_level0_ = model_ + model_config_size;
        model_higher_level_ = model_level0_ + level0_size;
        return true;
    }
    return false;
}
void Hnsw::SearchByVector(const vector<float>& qvec, size_t k, size_t ef_search, vector<pair<int, float>>& result) {
    if (model_ == nullptr) throw std::runtime_error("[Error] Model has not loaded!");
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* qraw = nullptr;

    if (ef_search < 0) {
        ef_search = 50 * k;
    }

    vector<float> qvec_copy(qvec);
    if(metric_ == DistanceKind::ANGULAR) {
        NormalizeVector(qvec_copy);
    }

    qraw = &qvec_copy[0];
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    int maxlevel = maxlevel_;
    int cur_node_id = enterpoint_id_;
    float cur_dist = dist_cls_->Evaluate(qraw, (float *)(model_level0_ + cur_node_id*memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes);
    float d;

    vector<pair<int, float> > path;
    if (ensure_k_) path.emplace_back(cur_node_id, cur_dist);

    bool changed;
    for (int i = maxlevel; i > 0; --i) {
        changed = true;
        while (changed) {
            changed = false;
            char* level_offset = model_level0_ + cur_node_id*memory_per_node_level0_;
            int offset = *((int*)(level_offset));
            char* level_base_offset = model_higher_level_ + offset * memory_per_node_higher_level_;
            int *data = (int*)(level_base_offset + (i-1) * memory_per_node_higher_level_);
            int size = *data;

            for (int j = 1; j <= size; ++j) {
                int tnum = *(data + j);
                d = (dist_cls_->Evaluate(qraw, (float *)(model_level0_ + tnum*memory_per_node_level0_ + memory_per_link_level0_), data_dim_, TmpRes));
                if (d < cur_dist) {
                    cur_dist = d;
                    cur_node_id = tnum;
                    offset = *((int*)(model_level0_ + cur_node_id*memory_per_node_level0_));
                    changed = true;
                    if (ensure_k_) path.emplace_back(cur_node_id, cur_dist);
                 }
            }
        }
    }

    if (ensure_k_) {
        while (result.size() < k && !path.empty()) {
            cur_node_id = path.back().first;
            cur_dist = path.back().second;
            path.pop_back();
            SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, result);
        }
    } else {
        SearchById_(cur_node_id, cur_dist, qraw, k, ef_search, result);
    }
}

void Hnsw::SearchById(int id, size_t k, size_t ef_search, vector<pair<int, float> >& result) {
    if (ef_search < 0) {
        ef_search = 50 * k;
    }
    SearchById_(id, 0.0, (const float*)(model_level0_ + id * memory_per_node_level0_ + memory_per_link_level0_), k, ef_search, result);
}

void Hnsw::SearchAtLayer(const std::vector<float>& qvec, HnswNode* enterpoint, int level, size_t ef, priority_queue<FurtherFirst>& result) {
    // TODO: check Node 12bytes => 8bytes
    _mm_prefetch(&dist_cls_, _MM_HINT_T0);
    float PORTABLE_ALIGN32 TmpRes[8];
    const float* qraw = &qvec[0];
    priority_queue<CloserFirst> candidates;
    float d = dist_cls_->Evaluate(qraw, (float*)&(enterpoint->GetData()[0]), data_dim_, TmpRes);
    result.emplace(enterpoint, d);
    candidates.emplace(enterpoint, d);

    visited_list_->Reset();
    unsigned int mark = visited_list_->GetVisitMark();
    unsigned int* visited = visited_list_->GetVisited();
    visited[enterpoint->GetId()] = mark;

    while(!candidates.empty()) {
        const CloserFirst& cand = candidates.top();
        float lowerbound = result.top().GetDistance();
        if (cand.GetDistance() > lowerbound) break;
        HnswNode* cand_node = cand.GetNode();
        unique_lock<mutex> lock(cand_node->access_guard_);
        const vector<HnswNode*>& neighbors = cand_node->GetFriends(level);
        candidates.pop();
        for (size_t j = 0; j < neighbors.size(); ++j) {
            _mm_prefetch((char*)&(neighbors[j]->GetData()), _MM_HINT_T0);
        }
        for (size_t j = 0; j < neighbors.size(); ++j) {
            int fid = neighbors[j]->GetId();
            if (visited[fid] != mark) {
                _mm_prefetch((char*)&(neighbors[j]->GetData()), _MM_HINT_T0);
                visited[fid] = mark;
                d = dist_cls_->Evaluate(qraw, (float*)&neighbors[j]->GetData()[0], data_dim_, TmpRes);
                if (result.size() < ef || result.top().GetDistance() > d) {
                    result.emplace(neighbors[j], d);
                    candidates.emplace(neighbors[j], d);
                    if (result.size() > ef) result.pop();
                }
            }
        }
    }
}

size_t Hnsw::GetModelConfigSize() const {
    size_t ret = 0;
    ret += sizeof(M_);
    ret += sizeof(MaxM_);
    ret += sizeof(MaxM0_);
    ret += sizeof(efConstruction_);
    ret += sizeof(levelmult_);
    ret += sizeof(maxlevel_);
    ret += sizeof(enterpoint_id_);
    ret += sizeof(num_nodes_);
    ret += sizeof(data_dim_);
    ret += sizeof(memory_per_data_);
    ret += sizeof(memory_per_link_level0_);
    ret += sizeof(memory_per_node_level0_);
    ret += sizeof(memory_per_node_higher_level_);
    ret += sizeof(higher_level_offset_);
    ret += sizeof(level0_offset_);
    return ret;
}

void Hnsw::SaveModelConfig(char* ptr) {
    ptr = SetValueAndIncPtr<size_t>(ptr, M_);
    ptr = SetValueAndIncPtr<size_t>(ptr, MaxM_);
    ptr = SetValueAndIncPtr<size_t>(ptr, MaxM0_);
    ptr = SetValueAndIncPtr<size_t>(ptr, efConstruction_);
    ptr = SetValueAndIncPtr<float>(ptr, levelmult_);
    ptr = SetValueAndIncPtr<int>(ptr, maxlevel_);
    ptr = SetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = SetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = SetValueAndIncPtr<DistanceKind>(ptr, metric_);
    ptr = SetValueAndIncPtr<size_t>(ptr, data_dim_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_data_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_link_level0_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_node_level0_);
    ptr = SetValueAndIncPtr<long long>(ptr, memory_per_node_higher_level_);
    ptr = SetValueAndIncPtr<long long>(ptr, higher_level_offset_);
    ptr = SetValueAndIncPtr<long long>(ptr, level0_offset_);
}

void Hnsw::PrintConfigs() const {
    logger_->info("HNSW configurations & status: M({}), MaxM({}), MaxM0({}), efCon({}), levelmult({}), maxlevel({}), #nodes({}), dimension of data({}), memory per data({}), memory per link level0({}), memory per node level0({}), memory per node higher level({}), higher level offset({}), level0 offset({})", M_, MaxM_, MaxM0_, efConstruction_, levelmult_, maxlevel_, num_nodes_, data_dim_, memory_per_data_, memory_per_link_level0_, memory_per_node_level0_, memory_per_node_higher_level_, higher_level_offset_, level0_offset_);
}

void Hnsw::PrintDegreeDist() const {
    logger_->info("* Degree distribution");
    vector<int> degrees(MaxM0_ + 2, 0);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        degrees[nodes_[i]->GetFriends(0).size()]++;
    }
    for (size_t i = 0; i < degrees.size(); ++i) {
        logger_->info("degree: {}, count: {}", i, degrees[i]);
    }
}

} // namespace n2
