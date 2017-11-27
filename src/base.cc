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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

#include "n2/base.h"


namespace n2 {

using std::string;
using std::vector;

Data::Data(const vector<float>& vec)
    :data_(vec) {
}

float GetTimeDiff(const std::chrono::steady_clock::time_point& begin_t,
                  const std::chrono::steady_clock::time_point& end_t) {
    return ((float)std::chrono::duration_cast<std::chrono::microseconds>(end_t - begin_t).count()) / 1000.0 / 1000.0;
}

string GetCurrentDateTime() {
    time_t now;
    time(&now);
    struct tm* timeinfo = localtime(&now);
    char time_string[50];
    strftime(time_string, sizeof(time_string), "%Y-%m-%d %H:%M:%S", timeinfo);
    return string(time_string);
}

} // namespace n2
