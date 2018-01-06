Go code example
===============

.. code:: go

    package main

    import (
        "n2"
        "fmt"
        "math/rand"
    )

    func main() {
         f := 3
         t := n2.NewHnswIndex(f)
         for i := 0; i < 1000; i++ {
           item := make([]float32, 0, f)
           for x:= 0; x < f; x++ {
               item = append(item, rand.Float32())
           }
           t.AddData(item)
         }
         t.Build(5, 10, 4, 10, 3.5, "heuristic", "skip")
         t.PrintConfigs()
         t.SaveModel("test.ann")
         var result []int
         var distance []float32
         t.SearchByVector([]float32{2, 1, 0}, 1000, -1, &result, &distance)
         fmt.Println(result)
         fmt.Println(distance)
    }

Go API
======

**Note that if a user passes a negative value it, will be set to a
default value when the metric has the default value.**

-  ``HnswIndex(dim, metric)``: returns a new Hnsw index

   -  ``dim`` (int): dimension of vectors
   -  ``metric`` (string): an optional parameter for choosing a metric
      of distance. (‘L2’\|'euclidean'\|‘angular’)

-  ``index.AddData(v)``: adds vector ``v``

   -  ``v`` (list of float): a vector with dimension ``dim``

-  ``index.Build(M, Max_M0, ef_construction, n_threads, mult, neighbor_selecting, graph_merging)``:
   builds a hnsw graph with given configurations.

   -  ``M`` (int): max number of edges for nodes at level>0 (default=12)
   -  ``Max_M0`` (int): max number of edges for nodes at level==0
      (default=24)
   -  ``ef_construction`` (int): efConstruction (see HNSW paper…)
      (default=100)
   -  ``n_threads`` (int): number of threads for building index
   -  ``mult`` (float): level multiplier (recommend: use default value)
      (default=1/log(1.0*M))
   -  ``neighbor_selecting`` (string): neighbor selecting policy

      -  available values

         -  ``"heuristic"``\ (default): select neighbors using
            algorithm4 on HNSW paper (recommended)
         -  ``"naive"``: select closest neighbors (not recommended)

   -  ``graph_merging`` (string): graph merging heuristic

      -  available values

         -  ``"skip"`` (default): do not merge (recommended for large
            scale of data(over 10M))
         -  ``"merge_level0"``: build an another graph in reverse order.
            then merge edges of level0 (recommended for under 10M scale
            data)

-  ``index.SaveModel(fname)``: saves the index to disk

   -  ``fname`` (string)

-  ``index.LoadModel(fname, use_mmap)``: loads an index from disk with
   mmap

   -  ``fname`` (string)
   -  ``use_mmap`` (bool): An optional parameter, default value is true.
      If this parameter is set, N2 loads model through mmap

-  ``index.UnloadModel()``: unloads (unmap)
-  ``index.SearchByVector(item_id, k, ef_serach=-1, vectors, distances)``:
   returns ``k`` nearest items.

   -  ``ef_search`` (int): default value = 50 \* k

-  ``index.SearchById(v, k, ef_serach=-1, vectors, distances)``: returns
   ``k`` nearest items.

   -  ``v`` (list of float): a query vector
   -  ``k`` (int)
   -  ``ef_search`` (int): default value = 50 \* k
