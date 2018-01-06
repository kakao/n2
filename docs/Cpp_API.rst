C++ code example
================

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
        

C++ API
=======

**Note that if a user passes a negative value, it will be set to a
default value when the metric has the default value.**

-  ``n2::HnswIndex(int dim, std::string metric = "angular")``: returns a
   new Hnsw index

   -  ``dim`` (int): dimension of vectors
   -  ``metric`` (std::string): an optional parameter for choosing a
      metric of distance. (‘L2’\|'euclidean'\|‘angular’). A default metric is
      ‘angular’.

-  ``void n2::HnswIndex.AddData(const std::vector<float>& data)``: adds
   vector ``v``

   -  ``data`` (std::vector): a vector with dimension ``dim``

-  ``void n2::HnswIndex.SetConfigs(const std::vector<std::pair<std::string, std::string> >& configs)``:
   Set configurations by key/value configures.

   -  ``M`` (int): max number of edges for nodes at level>0 (default=12)
   -  ``M0`` (int): max number of edges for nodes at level==0
      (default=24)
   -  ``ef_construction`` (int): efConstruction (see HNSW paper…)
      (default=150)
   -  ``n_threads`` (int): number of threads for building index
   -  ``mult`` (float): level multiplier (recommend: use default value)
      (default=1/log(1.0*M))
   -  ``neighbor_selecting`` (string): neighbor selecting policy
   -  available values

      -  ``NeighborSelectingPolicy::HEURISTIC``\ (default): select
         neighbors using algorithm4 on HNSW paper (recommended)
      -  ``NeighborSelectingPolicy::NAIVE``: select closest neighbors
         (not recommended)
      -  ``NeighborSelectingPolicy::HEURISTIC_SAVE_REMAINS``: explain is
         needed.

   -  ``graph_merging`` (string): graph merging heuristic
   -  available values

      -  ``GraphPostProcessing::SKIP`` (default): do not merge
         (recommended for large scale of data(over 10M))
      -  ``GraphPostProcessing::MERGE_LEVEL0``: build an another graph
         in reverse order. then merge edges of level0 (recommended for
         under 10M scale data)

-  ``void n2::HnswIndex.Fit()``: builds a hnsw graph with given
   configurations.
-  ``void Build(int M = -1, int M0 = -1, int ef_construction = -1, int n_threads = -1, float mult = -1, NeighborSelectingPolicy neighbor_selecting = NeighborSelectingPolicy::HEURISTIC, GraphPostProcessing graph_merging = GraphPostProcessing::SKIP, bool ensure_k = false)``:
   builds a hnsw graph with given configurations. (see ``Fit``,
   ``SetConfigs``)
-  ``bool n2::HnswIndex.SaveModel(const std::string& fname)``: saves the
   index to disk

   -  ``fname`` (std::string) : A index file name.

-  ``bool n2::HnswIndex.LoadModel(const std::string& fname, const bool use_mmap=true)``:
   loads an index from disk.

   -  ``fname`` (std::string) : A index file name.
   -  ``use_mmap``\ (bool): An optional parameter, default value is
      true. If this parameter is set, N2 loads model through mmap.

-  ``bool n2::HnswIndex.UnloadModel()``: Unload the loaded index file.
-  ``void n2::HnswIndex.SearchById(int id, size_t k, size_t ef_search, std::vector<std::pair<int, float> >& result)``:
   Returns ``k`` nearest items which are searched by Id.

   -  ``ef_search`` (int): default value = 50 \* k

-  ``void n2::HnswIndex.SearchByVector(const std::vector<float>& qvec, size_t k, size_t ef_search, std::vector<std::pair<int, float> >& result)``:
   Returns ``k`` nearest items which are searched by vector.

   -  ``ef_search`` (int): default value = 50 \* k

-  ``void n2::HnswIndex.PrintDegreeDist() const``: Print degree
   distributions.
-  ``void n2::HnswIndex.PrintConfigs() const``: Print index
   configurations.
