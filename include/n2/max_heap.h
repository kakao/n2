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

#include <boost/heap/d_ary_heap.hpp>

namespace n2 {

typedef typename std::pair<int, float> IdDistancePair;
struct IdDistancePairMaxHeapComparer {
	bool operator()(const IdDistancePair& p1, const IdDistancePair& p2) const {
        return p1.second < p2.second;
    }
};
typedef typename boost::heap::d_ary_heap<float, boost::heap::arity<4>> DistanceMaxHeap;

} // namespace n2
