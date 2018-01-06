Python code example
===================

.. code:: python

    from n2 import HnswIndex
    import random

    f = 40
    t = HnswIndex(f) # HnswIndex(f, "L2, euclidean, or angular") 
    for i in xrange(1000):
        v = [random.gauss(0, 1) for z in xrange(f)]
        t.add_data(v)

    t.build(m=5, max_m0=10, n_threads=4)
    t.save('test.hnsw')

    u = HnswIndex(f)
    u.load('test.hnsw')
    print(u.search_by_id(0, 1000))

Python API
==========

**Note that if a user passes a negative value, it will be set to a
default value when the metric has the default value.**

-  ``HnswIndex(dim, metric)``: returns a new Hnsw index

   -  ``dim`` (int): dimension of vectors
   -  ``metric`` (string): an optional parameter for choosing a metric
      of distance. (‘L2’\|'euclidean'\|‘angular’)

-  ``index.add_data(v)``: adds vector ``v``

   -  ``v`` (list of float): a vector with dimension ``dim``

-  ``index.build(m, max_m0, ef_construction, n_threads, mult, neighbor_selecting, graph_merging)``:
   builds a hnsw graph with given configurations.

   -  ``m`` (int): max number of edges for nodes at level>0 (default=12)
   -  ``max_m0`` (int): max number of edges for nodes at level==0
      (default=24)
   -  ``ef_construction`` (int): efConstruction (see HNSW paper…)
      (default=150)
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

-  ``index.save(fname)``: saves the index to a disk

   -  ``fname`` (string)

-  ``index.load(fname)``: loads an index from a disk.

   -  ``fname`` (string)  - ``use_mmap`` (bool): An optional parameter,
      default value is true. If this parameter is set, N2 loads model
      through mmap.

-  ``index.unload()``: unloads (unmap)
-  ``index.search_by_id(item_id, k, ef_serach=-1, include_distances=False)``:
   returns ``k`` nearest items.

   -  ``item_id`` (int)
   -  ``k`` (int)
   -  ``ef_search`` (int): default value = 50 \* k
   -  ``include_distances`` (boolean): If you set this argument to True,
      it will return a list of tuples(\ ``(item_id, distance)``).

-  ``index.search_by_vector(v, k, ef_serach=-1, include_distances=False)``:
   returns ``k`` nearest items.

   -  ``v`` (list of float): a query vector
   -  ``k`` (int)
   -  ``ef_search`` (int): default value = 50 \* k
   -  ``include_distances`` (boolean): If you set this argument to True,
      it will return a list of tuples(\ ``(item_id, distance)``).
