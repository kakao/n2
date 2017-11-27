# N2

N2 - approximate **N**earest **N**eighbor

```python
import numpy as np
from n2 import HnswIndex

N, dim = 10240, 20
samples = np.arange(N * dim).reshape(N, dim)

index = HnswIndex(dim)
for sample in samples:
    index.add_data(sample)
index.build(M=5, n_threads=4)
print(index.search_by_id(0, 10))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Introduce
N2 is an approximate nearest neighborhoods algorithm library written in C++ (including Python/Go bindings). N2 provides a much faster search speed than other implementations when modeling large dataset. Also, N2 supports multi-core CPUs for index building.

## Background
There are great approximate nearest neighborhoods libraries such as [annoy](https://github.com/spotify/annoy) and [nmslib](https://github.com/searchivarius/nmslib), but they did not fully meet the requirments to handle Kakao's dataset. Therefore, we decided to implement a library that improves usability and performs better based on [nmslib](https://github.com/searchivarius/nmslib). And finally, we release N2 to the world.

## Features
- Efficient implementations. N2 run faster and lightweights, especially for large dataset.
- Support multi-core CPUs for index building.
- Support a mmap feature by default for handling large index files efficiently.
- Support Python/Go bindings.

## Performance

If you want to read about detail benchmark explanation. See [the benchmark](docs/benchmark.md) for more experiments.

### Index build times
![](docs/imgs/build_time/build_time.png)

### Search speed
![](docs/imgs/search_time/search_speed.png)

### Memory usage

![](docs/imgs/mem/memory_usage.png)

## Install
See [the installation](INSTALL.md) for instruction on how to build N2 from source.

## Bindings
The following guides explain how to use N2 with basic examples and API.

- [Python](docs/Python_API.md)
- [C++](docs/Cpp_API.md)
- [Go](docs/Go_API.md)


## References
- "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Yu. A. Malkov, D. A. Yashunin (available on arxiv: http://arxiv.org/abs/1603.09320)
- nmslib: https://github.com/searchivarius/NMSLIB
- annoy: https://github.com/spotify/annoy


## License

This software is licensed under the [Apache 2 license](LICENSE.txt), quoted below.

Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this project except in compliance with the License. You may obtain a copy
of the License at http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

