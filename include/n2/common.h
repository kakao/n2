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

namespace n2 {

enum class GraphPostProcessing {
    SKIP = 0,
    MERGE_LEVEL0 = 1
};

enum class NeighborSelectingPolicy {
    NAIVE = 0,
    HEURISTIC = 1,
    HEURISTIC_SAVE_REMAINS = 2,
};

enum class DistanceKind {
    UNKNOWN = -1,
    ANGULAR = 0,
    L2 = 1
};

} // namespace n2
