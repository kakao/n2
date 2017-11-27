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

#include <unistd.h>

namespace n2 {
class Mmap {
    public:
    explicit Mmap(char const* fname);
    ~Mmap();
    void Map(char const* fname);
    void UnMap();
    size_t QueryFileSize() const;
    
    inline char* GetData() const { return data_; }
    inline bool IsOpen() const { return file_handle_ != 0; }
    inline int GetFileHandle() const { return file_handle_; }
    inline size_t GetFileSize() const { return file_size_; }

    private:
    char* data_ = nullptr;
    size_t file_size_ = 0;
    int file_handle_ = -1;
};
} // namespace n2
