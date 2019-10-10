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

#include <eigen3/Eigen/Dense>

namespace n2 
{
/*
template <typename F>
class Distance {
    public:
        Distance(F functor) : functor_(functor) {}
        inline float operator()(const float* v1, const float* v2, size_t qty) const {
            return functor_(v1, v2, qty);
        }
    private:
        F functor_;
};

class L2Distance {
    public:
        inline float operator()(const float* v1, const float* v2, size_t qty) const {
            Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> p(v1, qty, 1), q(v2, qty, 1);
            return (p - q).squaredNorm();
        }
};

class AngularDistance {
    public:
        inline float operator()(const float* v1, const float* v2, size_t qty) const {
            Eigen::Map<const Eigen::VectorXf, Eigen::Unaligned> p(v1, qty, 1), q(v2, qty, 1);
            return 1.0 - p.dot(q);
        }
};
*/

typedef float (*distance_function)(const float*, const float*, size_t);
float l2_distance(const float* v1, const float* v2, size_t qty);
float angular_distance(const float* v1, const float* v2, size_t qty);

} // namespace n2
