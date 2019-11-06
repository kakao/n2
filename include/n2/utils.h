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

#include <algorithm>
#include <numeric>
#include <vector>

namespace n2 {

class Utils {
public:
    static void NormalizeVector(const std::vector<float>& in, std::vector<float>& out) {
        float sum = std::inner_product(in.begin(), in.end(), in.begin(), 0.0);
        if (sum != 0.0) {
            sum = 1 / std::sqrt(sum);
            std::transform(in.begin(), in.end(), out.begin(), std::bind1st(std::multiplies<float>(), sum));
        }
    }
};

} // namespace n2
