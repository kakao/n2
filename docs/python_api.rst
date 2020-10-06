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
---------------------------------------------------------------------


.. autoclass:: n2.HnswIndex
   :members:

.. _examples/python: https://github.com/kakao/n2/tree/dev/examples/python
