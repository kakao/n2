Go Interface
==============================================================================

Basic Usage
------------------------------------------------------------------------------

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

You can see more code examples at `examples/go`_.

Main Interface
------------------------------------------------------------------------------

HnswIndex(dim, metric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Returns a new Hnsw index.
-  ``dim`` (int): Dimension of vectors.
-  ``metric`` (string): An optional parameter to choose a distance metric
   ('angular' | 'L2' | 'dot').

index.AddData(v)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Adds vector ``v``.
-  ``v`` (list of float): A vector with dimension ``dim``.

index.Build(M, Max_M0, ef_construction, n_threads, mult, neighbor_selecting, graph_merging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Builds a hnsw graph with given configurations.

-  ``M`` (int): Max number of edges for nodes at level > 0 (default: 12).
-  ``Max_M0`` (int): Max number of edges for nodes at level == 0
   (default: 24).
-  ``ef_construction`` (int): efConstruction (see HNSW paper.)
   (default: 100).
-  ``n_threads`` (int): Number of threads for building index.
-  ``mult`` (float): Level multiplier (recommended: use default value)
   (default: 1/log(1.0*M)).
-  ``neighbor_selecting`` (string): Neighbor selecting policy.

   -  Available values

      -  ``"heuristic"`` (default): Select neighbors using
         algorithm4 on HNSW paper (recommended).
      -  ``"naive"``: Select closest neighbors (not recommended).

-  ``graph_merging`` (string): Graph merging heuristic.

   -  Available values

      -  ``"skip"`` (default): Do not merge
         (recommended for large-scale data (over 10M)).
      -  ``"merge_level0"``: Performs an additional graph build in reverse order,
         then merges edges at level 0. So, it takes twice the build time
         compared to ``"skip"`` but shows slightly higher accuracy.
         (recommended for data under 10M scale).

index.SaveModel(fname)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Saves the index to disk.
-  ``fname`` (string)

index.LoadModel(fname, use_mmap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Loads an index from disk with mmap.
-  ``fname`` (string)
-  ``use_mmap`` (bool): An optional parameter (default: true).
   If this parameter is set, N2 loads model through mmap.

index.UnloadModel()
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Unloads (unmap) the index.

index.SearchByVector(item_id, k, ef_search=-1, vectors, distances)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Returns ``k`` nearest items (as vectors) to a query item.
-  ``ef_search`` (int): (default: 50 * k).

index.SearchById(v, k, ef_search=-1, vectors, distances)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-  Returns ``k`` nearest items (as ids) to a query item.
-  ``v`` (list of float): A query vector.
-  ``k`` (int)
-  ``ef_search`` (int): (default: 50 * k).

.. note::

   Currently, batch search functions are not supported in Go binding.

.. _examples/go: https://github.com/kakao/n2/tree/master/examples/go
