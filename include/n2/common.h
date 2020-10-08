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
/** @file */
namespace n2 {

/**
 * Graph merging heuristic.
 */
enum class GraphPostProcessing {
    SKIP = 0, /**< Do not merge (recommended for large-scale data (over 10M)). */
    MERGE_LEVEL0 = 1 /**< Performs an additional graph build in reverse order,
    then merges edges at level 0. So, it takes twice the build time compared to
    ``"skip"`` but shows slightly higher accuracy. (recommended for data under 10M scale). */
};

/**
 * Neighbor selecting policy.
 */
enum class NeighborSelectingPolicy {
    NAIVE = 0, /**< Select closest neighbors (not recommended). */
    HEURISTIC = 1, /**< Select neighbors using algorithm4 on HNSW paper (recommended). */
    HEURISTIC_SAVE_REMAINS = 2, /**< Experimental. */
};

enum class DistanceKind {
    UNKNOWN = -1,
    ANGULAR = 0,
    L2 = 1,
    DOT = 2
};

} // namespace n2
