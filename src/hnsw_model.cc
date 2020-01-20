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

#include "n2/hnsw_model.h"

#include <cstdint>
#include <cstring>
#include <fstream>

#include "n2/mmap.h"

namespace n2 {

using std::fstream;
using std::ifstream;
using std::make_shared;
using std::memset;
using std::ofstream;
using std::runtime_error;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;

shared_ptr<const HnswModel> HnswModel::GenerateModel(const vector<HnswNode*> nodes, int enterpoint_id, 
                                                     int max_m, int max_m0, DistanceKind metric, int max_level,
                                                     size_t data_dim) {
    return shared_ptr<const HnswModel>(
            new HnswModel(nodes, enterpoint_id, max_m, max_m0, metric, max_level, data_dim));
}

HnswModel::HnswModel(const vector<HnswNode*> nodes, int enterpoint_id, int max_m, int max_m0, DistanceKind metric,
                     int max_level, size_t data_dim)
        : enterpoint_id_(enterpoint_id), max_level_(max_level), data_dim_(data_dim), metric_(metric) {

    uint64_t total_level = 0;
    for (const auto& node : nodes) {
        total_level += node->GetLevel();
    }

    num_nodes_ = nodes.size();
    uint64_t model_config_size = GetConfigSize();
    memory_per_node_higher_level_ = sizeof(int) * (1 + max_m);  // "1" for saving num_links
    uint64_t higher_level_size = memory_per_node_higher_level_ * total_level;
    memory_per_data_ = sizeof(float) * data_dim_;
    memory_per_link_level0_ = sizeof(int) * (1 + 1 + max_m0);  // "1" for offset pos, "1" for saving num_links
    memory_per_node_level0_ = memory_per_link_level0_ + memory_per_data_;
    uint64_t level0_size = memory_per_node_level0_ * num_nodes_;

    model_byte_size_ = model_config_size + level0_size + higher_level_size;
    model_ = new char[model_byte_size_];
    if (model_ == nullptr)
        throw runtime_error("[Error] Fail to allocate memory for optimised index (size: "
                            + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");

    memset(model_, 0, model_byte_size_);
    model_level0_ = model_ + model_config_size;
    model_level0_node_base_offset_ = model_level0_ + memory_per_link_level0_;
    model_higher_level_ = model_level0_ + level0_size;


    SaveConfigToModel();
    int higher_offset = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        int level = nodes[i]->GetLevel();
        if (level > 0) {
            nodes[i]->CopyDataAndLevel0LinksToOptIndex(model_level0_ + i * memory_per_node_level0_, higher_offset);
            nodes[i]->CopyHigherLevelLinksToOptIndex(model_higher_level_ + 
                                                     memory_per_node_higher_level_ * higher_offset, 
                                                     memory_per_node_higher_level_);
            higher_offset += nodes[i]->GetLevel();
        } else {
            nodes[i]->CopyDataAndLevel0LinksToOptIndex(model_level0_ + i * memory_per_node_level0_, 0);
        }
    }
}

shared_ptr<const HnswModel> HnswModel::LoadModelFromFile(const string& fname, const bool use_mmap) {
    return shared_ptr<const HnswModel>(new HnswModel(fname, use_mmap));
}

HnswModel::HnswModel(const std::string& fname, const bool use_mmap) {
    if(!use_mmap) {
        ifstream in;
        in.open(fname, fstream::in|fstream::binary|fstream::ate);
        if(in.is_open()) {
            model_byte_size_ = in.tellg();
            in.seekg(0, fstream::beg);
            model_ = new char[model_byte_size_];
            if (model_ == nullptr)
                throw runtime_error("[Error] Fail to allocate memory for optimised index (size: "
                                    + to_string(model_byte_size_ / (1024 * 1024)) + " MBytes)");

            in.read(model_, model_byte_size_);
            in.close();
        } else {
            throw runtime_error("[Error] Failed to load model to file: " + fname+ " not found!");
        }
    } else {
        model_mmap_ = new Mmap(fname.c_str());
        model_byte_size_ = model_mmap_->GetFileSize();
        model_ = model_mmap_->GetData();
    }

    LoadConfigFromModel();
}

HnswModel::~HnswModel() {
    // unload model
    if (model_mmap_ != nullptr) {
        model_mmap_->UnMap();
        delete model_mmap_;
        model_mmap_ = nullptr;
        model_ = nullptr;
        model_higher_level_ = nullptr;
        model_level0_ = nullptr;
        model_level0_node_base_offset_ = nullptr;
    } else {
        delete[] model_;
        model_ = nullptr;
        model_higher_level_ = nullptr;
        model_level0_ = nullptr;
        model_level0_node_base_offset_ = nullptr;
    }
}

size_t HnswModel::GetConfigSize() {
    size_t ret = 0;
    ret += sizeof(size_t);                          // dummy for m_
    ret += sizeof(size_t);                          // dummy for max_m_
    ret += sizeof(size_t);                          // dummy for max_m0_
    ret += sizeof(size_t);                          // dummy for ef_construction_
    ret += sizeof(float);                           // dummy for level_mult_
    ret += sizeof(max_level_);
    ret += sizeof(enterpoint_id_);
    ret += sizeof(num_nodes_);
    ret += sizeof(metric_);
    ret += sizeof(data_dim_);
    ret += sizeof(memory_per_data_);
    ret += sizeof(memory_per_link_level0_);
    ret += sizeof(memory_per_node_level0_);
    ret += sizeof(memory_per_node_higher_level_);
    ret += sizeof(uint64_t);                        // dummy for higher_level_offset_
    ret += sizeof(uint64_t) ;                       // dummy for level0_offset_
    ret -= sizeof(metric_);                         // for old version bug 
    return ret;
}

void HnswModel::SaveConfigToModel() {
    char* ptr = model_;
    ptr += sizeof(size_t);                                  // dummy for m_
    ptr += sizeof(size_t);                                  // dummy for max_m_
    ptr += sizeof(size_t);                                  // dummy for max_m0_
    ptr += sizeof(size_t);                                  // dummy for ef_construction_
    ptr += sizeof(float);                                   // dummy for level_mult_
    ptr = SetValueAndIncPtr<int>(ptr, max_level_);
    ptr = SetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = SetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = SetValueAndIncPtr<DistanceKind>(ptr, metric_);
    ptr = SetValueAndIncPtr<size_t>(ptr, data_dim_);
    ptr = SetValueAndIncPtr<uint64_t>(ptr, memory_per_data_);
    ptr = SetValueAndIncPtr<uint64_t>(ptr, memory_per_link_level0_);
    ptr = SetValueAndIncPtr<uint64_t>(ptr, memory_per_node_level0_);
    ptr = SetValueAndIncPtr<uint64_t>(ptr, memory_per_node_higher_level_);
}

void HnswModel::LoadConfigFromModel() {
    char* ptr = model_;
    ptr += sizeof(size_t);                                  // dummy for m_
    ptr += sizeof(size_t);                                  // dummy for max_m_
    ptr += sizeof(size_t);                                  // dummy for max_m0_
    ptr += sizeof(size_t);                                  // dummy for ef_construction_
    ptr += sizeof(float);                                   // dummy for level_mult_
    ptr = GetValueAndIncPtr<int>(ptr, max_level_);
    ptr = GetValueAndIncPtr<int>(ptr, enterpoint_id_);
    ptr = GetValueAndIncPtr<int>(ptr, num_nodes_);
    ptr = GetValueAndIncPtr<DistanceKind>(ptr, metric_);
    if (metric_ != DistanceKind::ANGULAR and metric_ != DistanceKind::L2) {
        throw runtime_error("[Error] Unknown distance metric. metric");
    }
    auto data_dim_bak = data_dim_;
    ptr = GetValueAndIncPtr<size_t>(ptr, data_dim_);
    if (data_dim_bak > 0 && data_dim_ != data_dim_bak) {
        throw runtime_error("[Error] index dimension(" + to_string(data_dim_bak)
                            + ") != model dimension(" + to_string(data_dim_) + ")");
        data_dim_ = data_dim_bak;
    }
    ptr = GetValueAndIncPtr<uint64_t>(ptr, memory_per_data_);
    ptr = GetValueAndIncPtr<uint64_t>(ptr, memory_per_link_level0_);
    ptr = GetValueAndIncPtr<uint64_t>(ptr, memory_per_node_level0_);
    ptr = GetValueAndIncPtr<uint64_t>(ptr, memory_per_node_higher_level_);

    uint64_t level0_size = memory_per_node_level0_ * num_nodes_;
    uint64_t model_config_size = GetConfigSize();
    model_level0_ = model_ + model_config_size;
    model_level0_node_base_offset_ = model_level0_ + memory_per_link_level0_;
    model_higher_level_ = model_level0_ + level0_size;
}


bool HnswModel::SaveModelToFile(const string& fname) const {
    ofstream b_stream(fname.c_str(), fstream::out|fstream::binary);
    if (b_stream) {
        b_stream.write(model_, model_byte_size_);
        return (b_stream.good());
    } else {
        throw runtime_error("[Error] Failed to save model to file: " + fname);
    }
    return false;
}

} // namespace n2
