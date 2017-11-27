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

#if defined(__GNUC__)
  #define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
  #define PORTABLE_ALIGN32 __declspec(align(32))
#endif

namespace n2 {

class BaseDistance {
    public:
    BaseDistance() {}
    virtual ~BaseDistance() = 0;
    virtual float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const = 0;
};

class L2Distance : public BaseDistance {
   public:
   L2Distance() {}
   ~L2Distance() override {}
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

class AngularDistance : public BaseDistance {
   public:
   AngularDistance() {}
   ~AngularDistance() override {}
   float Evaluate(const float* __restrict pVect1, const float*  __restrict pVect2, size_t qty, float  *  __restrict TmpRes) const override;
};

} // namespace n2
