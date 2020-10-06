Python Interface
==============================================================================

Basic Usage
------------------------------------------------------------------------------

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

You can see more code examples at `examples/python`_.

Main Interface
------------------------------------------------------------------------------

.. autosummary::
    :nosignatures:

    n2.HnswIndex.add_data
    n2.HnswIndex.save
    n2.HnswIndex.load
    n2.HnswIndex.unload
    n2.HnswIndex.build
    n2.HnswIndex.search_by_vector
    n2.HnswIndex.search_by_id
    n2.HnswIndex.batch_search_by_vectors
    n2.HnswIndex.batch_search_by_ids
    n2.HnswIndex.print_degree_dist
    n2.HnswIndex.print_configs


.. autoclass:: n2.HnswIndex
   :members:

.. _examples/python: https://github.com/kakao/n2/tree/dev/examples/python
