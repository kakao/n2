CPP Interface
==============================================================================

Basic Usage
------------------------------------------------------------------------------

.. code:: cpp

    #include "hnsw.h"
    #include <vector>
    #include <map>

    int main() {
        n2::Hnsw index(3, "angular");
        index.AddData(std::vector<float>{0, 0, 1});
        index.AddData(std::vector<float>{0, 1, 0});
        index.AddData(std::vector<float>{0, 0, 1});
        std::vector<std::pair<std::string, std::string>> configs;
        int n_threads = 10;
        int M = 5;
        int MaxM0 = 10;
        index.Build(M, MaxM0, -1, n_threads);
        std::vector<std::pair<int, float> > result;
        int ef_search = 3*10;
        index.SearchByVector(std::vector<float>{3, 2, 1}, 3, ef_search, result);
        return 0;
    }

You can see more code examples at `examples/cpp`_.

Main Interface
------------------------------------------------------------------------------

.. doxygenclass:: n2::Hnsw
   :members: Hnsw, AddData, SetConfigs, Fit, Build, SaveModel, LoadModel,
             UnloadModel, SearchById, SearchByVector,
             PrintDegreeDist, PrintConfigs
   :undoc-members:


Full Reference
------------------------------------------------------------------------------
This is a full documentation of C++ implementation auto-generated from code comments.

.. toctree::
   :maxdepth: 2
   
   api/cpp_reference_root

.. _examples/cpp: https://github.com/kakao/n2/tree/dev/examples/cpp
