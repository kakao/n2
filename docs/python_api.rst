Python Interface
==============================================================================

Basic Usage
------------------------------------------------------------------------------

.. code:: python

    import random

    from n2 import HnswIndex

    f = 40
    t = HnswIndex(f)  # HnswIndex(f, "angular, L2, or dot")
    for i in range(1000):
        v = [random.gauss(0, 1) for z in range(f)]
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
    n2.HnswIndex.build
    n2.HnswIndex.save
    n2.HnswIndex.load
    n2.HnswIndex.unload
    n2.HnswIndex.search_by_vector
    n2.HnswIndex.search_by_id
    n2.HnswIndex.batch_search_by_vectors
    n2.HnswIndex.batch_search_by_ids

.. autoclass:: n2.HnswIndex
   :members: __init__, add_data, save, load, unload, build,
             search_by_vector, search_by_id,
             batch_search_by_vectors, batch_search_by_ids

.. _examples/python: https://github.com/kakao/n2/tree/master/examples/python
