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

#include <cmath>
#include <cstddef>
#include <functional>
#include <xmmintrin.h>

#include "n2/distance.h"

namespace n2 {

BaseDistance::~BaseDistance() {

}

float L2Distance::Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const {
    size_t qty4  = qty/4;
    size_t qty16 = qty/16;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4  * qty4;
    const float* pEnd3 = pVect1 + qty;

    __m128  diff, v1, v2;
    __m128  sum = _mm_set1_ps(0);
    
    while (pVect1 < pEnd1) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum  = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum  = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum  = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum  = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        
    }

    while (pVect1 < pEnd2) {
        v1   = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2   = _mm_loadu_ps(pVect2); pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum  = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    float res= TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    while (pVect1 < pEnd3) {
        float diff = *pVect1++ - *pVect2++; 
        res += diff * diff;
    }

    return res;
}

float AngularDistance::Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const {
#ifdef USE_AVX
    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4; 

    __m256  sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1); pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2); pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128  v1, v2;
    __m128  sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return  1.0f -sum;
    //return std::max(0.0f, 1 - std::max(float(-1), std::min(float(1), sum)));
#else
    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float* pEnd1 = pVect1 + 16 * qty16;
    const float* pEnd2 = pVect1 + 4 * qty4;

    __m128  v1, v2;
    __m128  sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1); pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2); pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return std::fmax(0.0f, 1 - std::fmax(float(-1), std::fmin(float(1), sum)));
#endif
}

} // namespace n2
