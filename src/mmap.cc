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
#include <fcntl.h>
#include <iostream>
#include <unistd.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdexcept>

#include "n2/mmap.h"

namespace n2 {

Mmap::Mmap(char const* fname) {
    Map(fname);
}
   
Mmap::~Mmap() {
    UnMap();
    if(file_handle_ != -1) {
        close(file_handle_);
        file_handle_ = -1;
    }
}

void Mmap::Map(char const* fname) {
    UnMap();
    if (fname == nullptr) throw std::runtime_error("[Error] Invalid file name received. (nullptr)");
    file_handle_ = open(fname, O_RDONLY);
    if (file_handle_ == -1) throw std::runtime_error("[Error] Failed to read file: " + std::string(fname));
    file_size_ = QueryFileSize();
    if (file_size_ <= 0) throw std::runtime_error("[Error] Memory mapping failed! (file_size==zero)");
    data_ = static_cast<char*>(mmap(0, file_size_, PROT_READ, MAP_SHARED, file_handle_, 0));
    if (data_ == MAP_FAILED) throw std::runtime_error("[Error] Memory mapping failed!");
}

void Mmap::UnMap() {
    if (data_ != nullptr) {
        int ret = munmap(const_cast<char*>(data_), file_size_);
        if (ret != 0) throw std::runtime_error("[Error] Memory unmapping failed!");
    }
    data_ = nullptr;
    file_size_ = 0;
    if(file_handle_ != -1) {
        close(file_handle_);
        file_handle_ = -1;
    }    
}

size_t Mmap::QueryFileSize() const {
    struct stat sbuf;
    if (fstat(file_handle_, &sbuf) == -1) {
        return 0;
    } else {
        return (size_t)sbuf.st_size;
    }
}

}
