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

namespace n2 {

class Data{
public:
    Data(const std::vector<float>& data) : data_(data) {}
    inline const std::vector<float>& GetData() const { return data_; };
    inline const float* GetRawData() const { return &data_[0]; };
private:
    std::vector<float> data_;
};

} // namespace n2
