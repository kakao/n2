C++ Interface
==============================================================================

Basic Usage
------------------------------------------------------------------------------

.. code:: cpp

    #include "hnsw.h"

    #include <utility>
    #include <vector>

    int main() {
        n2::Hnsw index(3, "angular");
        index.AddData(std::vector<float>{0, 0, 1});
        index.AddData(std::vector<float>{0, 1, 0});
        index.AddData(std::vector<float>{0, 0, 1});

        int n_threads = 4;
        int m = 5;
        int max_m0 = 10;
        int ef_construction = 50;
        index.Build(m, max_m0, ef_construction, n_threads);
        std::vector<std::pair<int, float>> result;
        int ef_search = 30;
        index.SearchByVector(std::vector<float>{3, 2, 1}, 3, ef_search, result);
        return 0;
    }

You can see more code examples at `examples/cpp`_.

Main Interface
------------------------------------------------------------------------------

.. doxygenclass:: n2::Hnsw
   :members: Hnsw, AddData, SaveModel, LoadModel, UnloadModel, Build, Fit, SetConfigs,
             SearchByVector, SearchById, BatchSearchByVectors, BatchSearchByIds
   :undoc-members:


Full Reference
------------------------------------------------------------------------------
This is a full documentation of C++ implementation auto-generated from code comments.

.. toctree::
   :maxdepth: 2
   
   api/cpp_reference_root

.. _examples/cpp: https://github.com/kakao/n2/tree/master/examples/cpp
