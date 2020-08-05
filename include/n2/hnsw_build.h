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
#include <queue>

#include "spdlog/spdlog.h"

#include "common.h"
#include "distance.h"
#include "heuristic.h"
#include "hnsw_model.h"
#include "visited_list.h"

namespace n2 {

class HnswBuild {
public:
    static std::unique_ptr<HnswBuild> GenerateBuilder(int dim, DistanceKind metric);
    HnswBuild(int dim, DistanceKind metric);
    virtual ~HnswBuild() {};

    HnswBuild(const HnswBuild&) = delete;
    void operator=(const HnswBuild&) = delete;

    void AddData(const std::vector<float>& data);
    void SetConfigs(const std::vector<std::pair<std::string, std::string>>& configs);
    std::shared_ptr<const HnswModel> Build(int m, int max_m0, int ef_construction, int n_threads, float mult, 
                                           NeighborSelectingPolicy neighbor_selecting, 
                                           GraphPostProcessing graph_merging);
    std::shared_ptr<const HnswModel> Build();

    void PrintDegreeDist() const;
    void PrintConfigs() const;

protected:
    void SetConfigs(int m, int max_m0, int ef_construction, int n_threads, float mult, 
                   NeighborSelectingPolicy neighbor_selecting, GraphPostProcessing graph_merging);


    int GetRandomNodeLevel();
    int GetRandomSeedPerThread();
    void BuildGraph(bool reverse);
   
    virtual void InitPolicies() = 0;
    virtual void InsertNode(HnswNode* qnode, VisitedList* visited_list) = 0;
    virtual void SearchAtLayer(HnswNode* qnode, const std::vector<HnswNode*>& enterpoints, int level, 
                               VisitedList* visited_list, std::priority_queue<FurtherFirst>& result) = 0;
    virtual void Link(HnswNode* source, HnswNode* target, int level) = 0;
    virtual void MergeEdgesOfTwoGraphs(const std::vector<HnswNode*>& another_nodes) = 0;

protected:
    std::shared_ptr<spdlog::logger> logger_;
    
    const std::string n2_signature = "TOROS_N2@N9R4";
    size_t m_ = 12;
    size_t max_m_ = 12;
    size_t max_m0_ = 24;
    size_t ef_construction_ = 150;
    float level_mult_ = 1 / log(1.0*m_);
    int n_threads_ = 1;
    NeighborSelectingPolicy neighbor_selecting_ = NeighborSelectingPolicy::HEURISTIC;
    NeighborSelectingPolicy post_neighbor_selecting_ = NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS;
    GraphPostProcessing post_graph_process_ = GraphPostProcessing::SKIP;
    
    int max_level_ = 0;
    HnswNode* enterpoint_ = nullptr;
    std::vector<Data> data_list_;
    std::vector<HnswNode*> nodes_;
    int num_nodes_ = 0;
    size_t data_dim_ = 0;
    DistanceKind metric_;

    mutable std::mutex max_level_guard_;
};

template<typename DistFuncType>
class HnswBuildImpl : public HnswBuild {
public:
    HnswBuildImpl(int dim, DistanceKind metric);
    ~HnswBuildImpl() override;

protected:
    void InitPolicies() override;
    void InsertNode(HnswNode* qnode, VisitedList* visited_list) override;
    void SearchAtLayer(HnswNode* qnode, const std::vector<HnswNode*>& enterpoint, int level, 
                       VisitedList* visited_list, std::priority_queue<FurtherFirst>& result) override;
    void Link(HnswNode* source, HnswNode* target, int level) override;
    void MergeEdgesOfTwoGraphs(const std::vector<HnswNode*>& another_nodes) override;

protected:
    bool is_naive_ = false;
    std::unique_ptr<BaseNeighborSelectingPolicies> selecting_policy_;
    std::unique_ptr<BaseNeighborSelectingPolicies> post_selecting_policy_;
            
    //std::make_unique<HeuristicNeighborSelectingPolicies>(true);


    DistFuncType dist_func_;
};

using HnswBuildAngular = HnswBuildImpl<AngularDistance>;
using HnswBuildL2 = HnswBuildImpl<L2Distance>;
using HnswBuildDot = HnswBuildImpl<DotDistance>;

} // namespace n2
