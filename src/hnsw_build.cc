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

#include "n2/hnsw_build.h"

#include <iostream>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <thread>

#include "n2/distance.h"
#include "n2/hnsw_node.h"
#include "n2/max_heap.h"
#include "n2/min_heap.h"
#include "n2/utils.h"

namespace n2 {

using std::defer_lock;
using std::make_unique;
using std::min;
using std::mt19937;
using std::mutex;
using std::numeric_limits;
using std::pair;
using std::priority_queue;
using std::runtime_error;
using std::shared_ptr;
using std::stof;
using std::stoi;
using std::string;
using std::to_string;
using std::uniform_real_distribution;
using std::unique_lock;
using std::unique_ptr;
using std::unordered_set;
using std::vector;

unique_ptr<HnswBuild> HnswBuild::GenerateBuilder(int dim, DistanceKind metric) {
    if (metric == DistanceKind::ANGULAR) {
        return make_unique<HnswBuildAngular>(dim, metric);
    } else if (metric == DistanceKind::L2) {
        return make_unique<HnswBuildL2>(dim, metric);
    } else {
        throw runtime_error("[Error] Invalid configuration value for DistanceMethod");
    }
}

HnswBuild::HnswBuild(int dim, DistanceKind metric) : data_dim_(dim), metric_(metric) {
}

void HnswBuild::AddData(const vector<float>& data) {
    if (data.size() != data_dim_)
        throw runtime_error("[Error] Invalid dimension data inserted: " + to_string(data.size()) + 
                                 ", Predefined dimension: " + to_string(data_dim_));
    if (metric_ == DistanceKind::ANGULAR) {
        vector<float> normalized(data_dim_);
        Utils::NormalizeVector(data, normalized);
        data_list_.emplace_back(normalized);
    } else {
        data_list_.emplace_back(data);
    }
}

void HnswBuild::SetConfigs(const vector<pair<string, string>>& configs) {
    int m = -1, max_m0 = -1, ef_construction = -1, n_threads = -1;
    float mult = -1;
    NeighborSelectingPolicy neighbor_selecting = NeighborSelectingPolicy::HEURISTIC;
    GraphPostProcessing graph_merging = GraphPostProcessing::SKIP;

    for (const auto& c : configs) {
        if (c.first == "M") {
            m = stoi(c.second);
        } else if (c.first == "MaxM0") {
            max_m0 = stoi(c.second);
        } else if (c.first == "efConstruction") {
            ef_construction = stoi(c.second);
        } else if (c.first == "NumThread") {
            n_threads = stoi(c.second);
        } else if (c.first == "Mult") {
            mult = stof(c.second);
        } else if (c.first == "NeighborSelecting") {
            if (c.second == "heuristic") {
                neighbor_selecting = NeighborSelectingPolicy::HEURISTIC;
            } else if (c.second == "heuristic_save_remains") {
                neighbor_selecting = NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS;
            } else if (c.second == "naive") {
                neighbor_selecting = NeighborSelectingPolicy::NAIVE;
            } else {
                throw runtime_error("[Error] Invalid configuration value for NeighborSelecting: " + c.second);
            }
        } else if (c.first == "GraphMerging") {
            if (c.second == "skip") {
                graph_merging = GraphPostProcessing::SKIP;
            } else if (c.second == "merge_level0") {
                graph_merging = GraphPostProcessing::MERGE_LEVEL0;
            } else {
                throw runtime_error("[Error] Invalid configuration value for GraphMerging: " + c.second);
            }
        } else if (c.first == "EnsureK") {
        } else {
            throw runtime_error("[Error] Invalid configuration key: " + c.first);
        }
    }

    SetConfigs(m, max_m0, ef_construction, n_threads, mult, neighbor_selecting, graph_merging);
}

shared_ptr<const HnswModel> HnswBuild::Build(int m, int max_m0, int ef_construction, int n_threads, float mult, 
                                             NeighborSelectingPolicy neighbor_selecting, 
                                             GraphPostProcessing graph_merging) {
    SetConfigs(m, max_m0, ef_construction, n_threads, mult, neighbor_selecting, graph_merging);
    return Build();
}

shared_ptr<const HnswModel> HnswBuild::Build() {
    if (data_list_.size() == 0) 
        throw runtime_error("[Error] No data to fit. Load data first.");
    InitPolicies();
    BuildGraph(false);
    if (post_graph_process_ == GraphPostProcessing::MERGE_LEVEL0) {
        logger_->info("graph post processing: merge_level0");
        
        vector<HnswNode*> nodes_backup;
        nodes_backup.swap(nodes_);
        BuildGraph(true);
        MergeEdgesOfTwoGraphs(nodes_backup);
        
        for (size_t i = 0; i < nodes_backup.size(); ++i) {
            delete nodes_backup[i];
        }
        nodes_backup.clear();
    }

    auto&& model = HnswModel::GenerateModel(nodes_, enterpoint_->GetId(), max_m_, max_m0_, metric_, 
                                            max_level_, data_dim_);
    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }
    nodes_.clear();
    data_list_.clear();

    return model;
}

void HnswBuild::BuildGraph(bool reverse) {
    nodes_.resize(data_list_.size());
    int level = GetRandomNodeLevel();
    HnswNode* first = new HnswNode(0, &(data_list_[0]), level, max_m_, max_m0_);
    nodes_[0] = first;
    max_level_ = level;
    enterpoint_ = first;

    #pragma omp parallel num_threads(n_threads_) 
    {
        VisitedList* visited_list = new VisitedList(data_list_.size());
        if (reverse) {
            #pragma omp for schedule(dynamic,128)
            for (size_t i = data_list_.size() - 1; i >= 1; --i) {
                int level = GetRandomNodeLevel();
                HnswNode* qnode = new HnswNode(i, &data_list_[i], level, max_m_, max_m0_);
                nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
        } else {
            #pragma omp for schedule(dynamic,128)
            for (size_t i = 1; i < data_list_.size(); ++i) {
                int level = GetRandomNodeLevel();
                HnswNode* qnode = new HnswNode(i, &data_list_[i], level, max_m_, max_m0_);
                nodes_[i] = qnode;
                InsertNode(qnode, visited_list);
            }
        }
        delete visited_list;
    }
}

int HnswBuild::GetRandomNodeLevel() {
    static thread_local mt19937 rng(GetRandomSeedPerThread());
    static thread_local uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    double r = uniform_distribution(rng);

    if (r < numeric_limits<double>::epsilon())
        r = 1.0;
    return (int)(-log(r) * level_mult_);
}

int HnswBuild::GetRandomSeedPerThread() {
    int tid = omp_get_thread_num();
    int g_seed = 17;
    for (int i = 0; i <= tid; ++i)
        g_seed = 214013 * g_seed + 2531011;
    return (g_seed >> 16) & 0x7FFF;
}

void HnswBuild::SetConfigs(int m, int max_m0, int ef_construction, int n_threads, float mult, 
                          NeighborSelectingPolicy neighbor_selecting, GraphPostProcessing graph_merging) {
    if ( m > 0 ) max_m_ = m_ = m;
    if ( max_m0 > 0 ) max_m0_ = max_m0;
    if ( ef_construction > 0 ) ef_construction_ = ef_construction;
    if ( n_threads > 0 ) n_threads_ = n_threads;
    level_mult_ = mult > 0 ? mult : 1 / log(1.0 * m_);
    neighbor_selecting_ = neighbor_selecting;
    post_graph_process_ = graph_merging;
}

void HnswBuild::PrintDegreeDist() const {
    // TODO: not implemented yet
}

void HnswBuild::PrintConfigs() const {
    // TODO: not implemented yet
}

template<typename DistFuncType>
HnswBuildImpl<DistFuncType>::HnswBuildImpl(int dim, DistanceKind metric) : HnswBuild(dim, metric) {
    logger_ = spdlog::get("n2");
    if (logger_ == nullptr) {
        logger_ = spdlog::stdout_logger_mt("n2");
    }
}

template<typename DistFuncType>
HnswBuildImpl<DistFuncType>::~HnswBuildImpl() {
    for (size_t i = 0; i < nodes_.size(); ++i) {
        delete nodes_[i];
    }
}

template<typename DistFuncType>
void HnswBuildImpl<DistFuncType>::InitPolicies() {
    if (neighbor_selecting_ == NeighborSelectingPolicy::HEURISTIC) {
        selecting_policy_ = make_unique<HeuristicNeighborSelectingPolicies<DistFuncType>>(false);
        is_naive_ = false;
    } else if (neighbor_selecting_ == NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS) {
        selecting_policy_ = make_unique<HeuristicNeighborSelectingPolicies<DistFuncType>>(true);
        is_naive_ = false;
    } else if (neighbor_selecting_ == NeighborSelectingPolicy::NAIVE) {
        selecting_policy_ = make_unique<NaiveNeighborSelectingPolicies>();
        is_naive_ = true;
    }
    if (post_neighbor_selecting_ == NeighborSelectingPolicy::HEURISTIC) {
        post_selecting_policy_ = make_unique<HeuristicNeighborSelectingPolicies<DistFuncType>>(false);
    } else if (post_neighbor_selecting_ == NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS) {
        post_selecting_policy_ = make_unique<HeuristicNeighborSelectingPolicies<DistFuncType>>(true);
    } else if (post_neighbor_selecting_ == NeighborSelectingPolicy::NAIVE) {
        post_selecting_policy_ = make_unique<NaiveNeighborSelectingPolicies>();
    }
}

template<typename DistFuncType>
void HnswBuildImpl<DistFuncType>::InsertNode(HnswNode* qnode, VisitedList* visited_list) {
    int cur_level = qnode->GetLevel();
    unique_lock<mutex> max_level_lock(max_level_guard_, defer_lock);
    if (cur_level > max_level_)
        max_level_lock.lock();

    int max_level_copy = max_level_;
    HnswNode* enterpoint = enterpoint_;

    if (cur_level < max_level_copy) {
        HnswNode* cur_node = enterpoint;
        float d = dist_func_(qnode, cur_node, data_dim_);
        float cur_dist = d;
        for (auto i = max_level_copy; i > cur_level; --i) {
            bool changed = true;
            while (changed) {
                changed = false;
                unique_lock<mutex> local_lock(cur_node->GetAccessGuard());
                const vector<HnswNode*>& neighbors = cur_node->GetFriends(i);
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    _mm_prefetch((*iter)->GetData(), _MM_HINT_T0);
                }
                for (auto iter = neighbors.begin(); iter != neighbors.end(); ++iter) {
                    d = dist_func_(qnode, *iter, data_dim_);
                    if (d < cur_dist) {
                        cur_dist = d;
                        cur_node = *iter;
                        changed = true;
                    }
                }
            }
        }
        enterpoint = cur_node;
    }
    
    _mm_prefetch(&selecting_policy_, _MM_HINT_T0);
    for (auto i = min(max_level_copy, cur_level); i >= 0; --i) {
        priority_queue<FurtherFirst> result;
        SearchAtLayer(qnode, enterpoint, i, visited_list, result);
        selecting_policy_->Select(m_, data_dim_, result);
        while (result.size() > 0) {
            auto* top_node = result.top().GetNode();
            result.pop();
            Link(top_node, qnode, i);
            Link(qnode, top_node, i);
        }
    }
    if (cur_level > enterpoint_->GetLevel()) {
        enterpoint_ = qnode;
        max_level_ = cur_level;
    }
}

template<typename DistFuncType>
void HnswBuildImpl<DistFuncType>::SearchAtLayer(HnswNode* qnode, HnswNode* enterpoint, int level, 
                                                VisitedList* visited_list, priority_queue<FurtherFirst>& result) {
    // TODO: check Node 12bytes => 8bytes
    priority_queue<CloserFirst> candidates;
    float d = dist_func_(qnode, enterpoint, data_dim_);
    result.emplace(enterpoint, d);
    candidates.emplace(enterpoint, d);
    
    visited_list->Reset();
    visited_list->MarkAsVisited(enterpoint->GetId());
   
    while (!candidates.empty()) {
        const CloserFirst& candidate = candidates.top();
        float lower_bound = result.top().GetDistance();
        if (candidate.GetDistance() > lower_bound) 
            break;
        
        HnswNode* candidate_node = candidate.GetNode();
        unique_lock<mutex> lock(candidate_node->GetAccessGuard());
        const vector<HnswNode*>& neighbors = candidate_node->GetFriends(level);
        candidates.pop();
        for (const auto& neighbor : neighbors) {
            _mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
        }
        for (const auto& neighbor : neighbors) {
            int id = neighbor->GetId();
            if (visited_list->NotVisited(id)) {
                _mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
                visited_list->MarkAsVisited(id);
                d = dist_func_(qnode, neighbor, data_dim_);
                if (result.size() < ef_construction_ || result.top().GetDistance() > d) {
                    result.emplace(neighbor, d);
                    candidates.emplace(neighbor, d);
                    if (result.size() > ef_construction_)
                        result.pop();
                }
            }
        }
    }
}

template<typename DistFuncType>
void HnswBuildImpl<DistFuncType>::Link(HnswNode* source, HnswNode* target, int level) {
    unique_lock<mutex> lock(source->GetAccessGuard());
    vector<HnswNode*>& neighbors = source->GetFriends(level);
    neighbors.push_back(target);
    bool shrink = (level > 0 && neighbors.size() > source->GetMaxM()) ||
                  (level <= 0 && neighbors.size() > source->GetMaxM0());
    if (!shrink) return;
    if (is_naive_) {
        float max = dist_func_(source, neighbors[0], data_dim_);
        int maxi = 0;
        for (size_t i = 1; i < neighbors.size(); ++i) {
                float curd = dist_func_(source, neighbors[i], data_dim_);
                if (curd > max) {
                    max = curd;
                    maxi = i;
                }
        }
        neighbors.erase(neighbors.begin() + maxi);
    } else {
        priority_queue<FurtherFirst> tempres;
        for (const auto& neighbor : neighbors) {
            _mm_prefetch(neighbor->GetData(), _MM_HINT_T0);
        }

        for (const auto& neighbor : neighbors) {
            tempres.emplace(neighbor, dist_func_(source, neighbor, data_dim_));
        }
        selecting_policy_->Select(tempres.size() - 1, data_dim_, tempres);
        neighbors.clear();
        while (tempres.size()) {
            neighbors.emplace_back(tempres.top().GetNode());
            tempres.pop();
        }
    }
}

template<typename DistFuncType>
void HnswBuildImpl<DistFuncType>::MergeEdgesOfTwoGraphs(const vector<HnswNode*>& another_nodes) {
    #pragma omp parallel for schedule(dynamic,128) num_threads(n_threads_)
    for (size_t i = 1; i < data_list_.size(); ++i) {
        const vector<HnswNode*>& neighbors1 = nodes_[i]->GetFriends(0);
        const vector<HnswNode*>& neighbors2 = another_nodes[i]->GetFriends(0);
        unordered_set<int> merged_neighbor_id_set = unordered_set<int>();
        for (const auto& cur : neighbors1) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        for (const auto& cur : neighbors2) {
            merged_neighbor_id_set.insert(cur->GetId());
        }
        priority_queue<FurtherFirst> temp_res;
        for (int cur : merged_neighbor_id_set) {
            temp_res.emplace(nodes_[cur], dist_func_(data_list_[cur].GetRawData(), 
                                                     data_list_[i].GetRawData(), data_dim_));
        }

        // Post Heuristic
        post_selecting_policy_->Select(max_m0_, data_dim_, temp_res);
        vector<HnswNode*> merged_neighbors;
        while (!temp_res.empty()) {
            merged_neighbors.emplace_back(temp_res.top().GetNode());
            temp_res.pop();
        }
        nodes_[i]->SetFriends(0, merged_neighbors);
    }
}

template class HnswBuildImpl<AngularDistance>;
template class HnswBuildImpl<L2Distance>;

} // namespace n2
