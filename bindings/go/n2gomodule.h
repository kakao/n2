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

#include "n2/hnsw.h"

// IMPORTANT: Please write #cgo CXXFLAGS: -std=c++11 -lgomp -I./third_party/spdlog/include 
// on the hnswindex.go after generating swig files.

namespace n2 {
    class HnswIndex {
    public:
        HnswIndex(int dimension) {
            ptr = std::unique_ptr<Hnsw>(new Hnsw(dimension));
        };

        HnswIndex(int dimension, const char* metric) {
            ptr = std::unique_ptr<Hnsw>(new Hnsw(dimension, metric));
        };

        ~HnswIndex() {};
        
        void Build(int M, int Max_M0, int ef_construction, int n_threads, float mult, const char* neighbor_selecting, const char* graph_merging) {
            std::vector<std::pair<std::string, std::string>> configs;
            configs.emplace_back("M", std::to_string(M));
            configs.emplace_back("MaxM0", std::to_string(Max_M0));
            configs.emplace_back("efConstruction", std::to_string(ef_construction));
            configs.emplace_back("NumThread", std::to_string(n_threads));
            configs.emplace_back("Mult", std::to_string(mult));
            configs.emplace_back("NeighborSelecting", std::string(neighbor_selecting));
            configs.emplace_back("GraphMerging", std::string(graph_merging));
            ptr->SetConfigs(configs);
            ptr->Fit();
        };

        bool SaveModel(const char* fname) {
            std::string filename(fname);
            return ptr->SaveModel(filename);
        };

        bool LoadModel(const char* fname) {
            std::string filename(fname);
            return ptr->LoadModel(filename);
        };


        bool LoadModel(const char* fname, bool use_mmap) {
            std::string filename(fname);
            return ptr->LoadModel(filename, use_mmap);
        };

        void UnloadModel() {
            ptr->UnloadModel();
        };

        void AddData(std::vector<float> data) {
            ptr->AddData(data);
        };

        void SearchByVector(std::vector<float> vecs, int k, int ef_search,
                            std::vector<int>* ids) {
            std::vector<std::pair<int, float>> result_tmp;
            ptr->SearchByVector(vecs, k, ef_search, result_tmp);
            for(auto iter = result_tmp.begin(); iter != result_tmp.end(); iter++) {
                ids->emplace_back(iter->first);
            }
        };

        void SearchByVector(std::vector<float> vecs, int k, int ef_search,
                            std::vector<int>* ids, std::vector<float>* distances) {
            std::vector<std::pair<int, float>> result_tmp;
            ptr->SearchByVector(vecs, k, ef_search, result_tmp);
            for(auto iter = result_tmp.begin(); iter != result_tmp.end(); iter++) {
                ids->emplace_back(iter->first);
                distances->emplace_back(iter->second);
            }
        };

        void SearchById(int id, int k, int ef_search,
                        std::vector<int>* ids) {
            std::vector<std::pair<int, float>> result_tmp;
            ptr->SearchById(id, k, ef_search, result_tmp);
            for(auto iter = result_tmp.begin(); iter != result_tmp.end(); iter++) {
                ids->emplace_back(iter->first);
            }
            
        };

        void SearchById(int id, int k, int ef_search,
                          std::vector<int>* ids, std::vector<float>* distances) {
              std::vector<std::pair<int, float>> result_tmp;
              ptr->SearchById(id, k, ef_search, result_tmp);
              for(auto iter = result_tmp.begin(); iter != result_tmp.end(); iter++) {
                  ids->emplace_back(iter->first);
                  distances->emplace_back(iter->second);
              }
        };

        void PrintDegreeDist() {
            ptr->PrintDegreeDist();
        };

        void PrintConfigs() {
            ptr->PrintConfigs();
        };

    private:
        std::unique_ptr<Hnsw> ptr;
    };
}
