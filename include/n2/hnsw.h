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

#include <omp.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "hnsw_build.h"
#include "hnsw_model.h"
#include "hnsw_search.h"

namespace n2 {

class Hnsw {
public:
    Hnsw();
    
    /** @brief Makes an Hnsw Index instance.
     * @return a new Hnsw index
     */
    Hnsw(int dim, /**< dimenstion of vectors. */
         std::string metric="angular" /**< an optional parameter for choosing a metric of distance. ('L2'\|;'euclidean'\|'angular'). A default metric is 'angular'. */);
    Hnsw(const Hnsw& other);
    Hnsw(Hnsw&& other) noexcept;
    ~Hnsw();

    Hnsw& operator=(const Hnsw& other);
    Hnsw& operator=(Hnsw&& other) noexcept;

    ////////////////////////////////////////////
    // Build

    /** @brief Adds vector v. */
    void AddData(const std::vector<float>& data /**< A vector with dimension ``dim``. */);

    /** @brief Set configurations by key/value configures. */
    void SetConfigs(const std::vector<std::pair<std::string, std::string>>& configs);
    
    /**
     * @brief Builds a hnsw graph with given configurations.
     * @note Refer to `common.h` to see other available values for neighbor_selecting and graph_merging.
     * @see Fit(), SetConfigs()
     */
    void Build(int m=-1, /**< Max number of edges for nodes at level > 0 (default = 12) */
               int max_m0=-1, /**< Max number of edges for nodes at level == 0 (default = 24). */
               int ef_construction=-1, /**< Refer to HNSW paper for its role(default = 150). */
               int n_threads=-1, /**< Number of threads for building index. */
               float mult=-1, /**< Level multiplier (default value recommended) (default = 1/log(1.0*M)). */
               NeighborSelectingPolicy neighbor_selecting=NeighborSelectingPolicy::HEURISTIC, /**< Neighbor selecting policy (default = NeighborSelectingPolicy::HEURISTIC). */
               GraphPostProcessing graph_merging=GraphPostProcessing::SKIP, /**< Graph merging heuristic (default = GraphPostProcessing::SKIP). */
               bool ensure_k=false);

    /** @brief Builds a hnsw graph with given configurations. */
    void Fit();

    ////////////////////////////////////////////
    // Model
    /** @brief Saves the index to disk. */
    bool SaveModel(const std::string& fname /**< An index file name. */) const;

    /** @brief Loads an index from disk. */
    bool LoadModel(const std::string& fname, /**< An index file name. */ 
                   const bool use_mmap=true /**< An optional parameter (default = true). If this parameter is set, N2 loads model through mmap. */
                  );

    /** @brief Unloads the loaded index file. */
    void UnloadModel();
 
    ////////////////////////////////////////////
    // Search 
    inline void SearchByVector(const std::vector<float>& qvec, size_t k, size_t ef_search,
                               std::vector<int>& result) {
        searcher_->SearchByVector(qvec, k, ef_search, ensure_k_, result);
    }

    /** 
     * @return ``k`` nearest items which are searched by vector.
     */
    inline void SearchByVector(const std::vector<float>& qvec, size_t k, 
                               size_t ef_search, /**< (default = 50 * k) */
                               std::vector<std::pair<int, float>>& result) {
        searcher_->SearchByVector(qvec, k, ef_search, ensure_k_, result);
    }
    inline void SearchById(int id, size_t k, size_t ef_search, std::vector<int>& result) {
        searcher_->SearchById(id, k, ef_search, ensure_k_, result);
    }

    /**
     * @return ``k`` nearest items which are searched by Id.
     */
    inline void SearchById(int id, size_t k, size_t ef_search, /**< (default = 50 * k). */
        std::vector<std::pair<int, float>>& result) {
        searcher_->SearchById(id, k, ef_search, ensure_k_, result);
    }

    inline void BatchSearchByVectors(const std::vector<std::vector<float>>& qvecs, size_t k, 
                                     size_t ef_search, size_t n_threads, std::vector<std::vector<int>>& results) {
        BatchSearchByVectors_(qvecs, k, ef_search, n_threads, results);
    }
    inline void BatchSearchByVectors(const std::vector<std::vector<float>>& qvecs, size_t k, 
                                     size_t ef_search, size_t n_threads, 
                                     std::vector<std::vector<std::pair<int, float>>>& results) {
        BatchSearchByVectors_(qvecs, k, ef_search, n_threads, results);
    }
    inline void BatchSearchByIds(const std::vector<int> ids, size_t k, size_t ef_search, size_t n_threads,
                                 std::vector<std::vector<int>>& results) {
        BatchSearchByIds_(ids, k, ef_search, n_threads, results);
    }
    inline void BatchSearchByIds(const std::vector<int> ids, size_t k, size_t ef_search, size_t n_threads,
                                 std::vector<std::vector<std::pair<int, float>>>& results) {
        BatchSearchByIds_(ids, k, ef_search, n_threads, results);
    }

    ////////////////////////////////////////////
    // Build(Misc)
    /** @brief Prints degree distributions. */
    void PrintDegreeDist() const;

    /** @brief Prints index configurations. */
    void PrintConfigs() const;

private:
    void InitSearcherAndSearcherPool_();

    template<typename ResultType>
    void BatchSearchByVectors_(const std::vector<std::vector<float>>& qvecs, size_t k, 
                               size_t ef_search, size_t n_threads, ResultType& results) {
        results.resize(qvecs.size());
        while (searcher_pool_.size() < n_threads) {
            searcher_pool_.push_back(HnswSearch::GenerateSearcher(model_, data_dim_, metric_));
        }

        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < qvecs.size(); ++i) {
                auto& s = searcher_pool_[omp_get_thread_num()];
                s->SearchByVector(qvecs[i], k, ef_search, ensure_k_, results[i]);
            }
        }
    }

    template<typename ResultType>
    void BatchSearchByIds_(const std::vector<int> ids, size_t k, size_t ef_search, size_t n_threads,
                           ResultType& results) {
        results.resize(ids.size());
        while (searcher_pool_.size() < n_threads) {
            searcher_pool_.push_back(HnswSearch::GenerateSearcher(model_, data_dim_, metric_));
        }

        #pragma omp parallel num_threads(n_threads)
        {
            #pragma omp for schedule(runtime)
            for (size_t i = 0; i < ids.size(); ++i) {
                auto& s = searcher_pool_[omp_get_thread_num()];
                s->SearchById(ids[i], k, ef_search, ensure_k_, results[i]);
            }
        }
    }

private:
    std::unique_ptr<HnswBuild> builder_;
    std::shared_ptr<const HnswModel> model_;
    std::shared_ptr<HnswSearch> searcher_;                      // for single-thread search
    std::vector<std::shared_ptr<HnswSearch>> searcher_pool_;    // for multi-threads batch search

    size_t data_dim_;
    DistanceKind metric_;
    bool ensure_k_ = false;
};

} // namespace n2
