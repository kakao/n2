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

class VisitedList { 
public:
    VisitedList(int size) : size_(size), mark_(1) {
        visited_ = new unsigned int[size_];
        memset(visited_, 0, sizeof(unsigned int)*size_);
    }
    
    ~VisitedList() { delete [] visited_; }
    
    inline bool Visited(unsigned int index) const { return visited_[index] == mark_; }
    inline bool NotVisited(unsigned int index) const { return visited_[index] != mark_; }
    inline void MarkAsVisited(unsigned int index) { visited_[index] = mark_; }
    inline void Reset() {
        if (++mark_ == 0) {
            mark_ = 1;
            memset(visited_, 0, sizeof(unsigned int)*size_);
        }
    }
    inline unsigned int* GetVisited() { return visited_; }
    inline unsigned int GetVisitMark() { return mark_; }

private:
    unsigned int* visited_;
    unsigned int size_;
    unsigned int mark_;
};

} // namespace n2
