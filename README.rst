N2
==============================================================================

|pypi| |docs| |travis| |license|

.. begin_badges

.. |docs| image:: https://readthedocs.org/projects/n2/badge/?version=latest
   :target: https://n2.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |pypi| image:: https://img.shields.io/pypi/v/n2.svg?style=flat
   :target: https://pypi.python.org/pypi/n2
   :alt: Latest Version

.. |travis| image:: https://travis-ci.org/kakao/n2.svg?branch=master
   :target: https://travis-ci.org/kakao/n2
   :alt: Build Status

.. |license| image:: https://img.shields.io/github/license/kakao/n2
   :target: https://github.com/kakao/n2/blob/master/LICENSE
   :alt: Apache-License 2.0

.. end_badges

.. begin_intro

Lightweight approximate **N**\ earest **N**\ eighbor algorithm library written
in C++ (with Python/Go bindings).

N2 stands for two N's, which comes from \'Approximate ``N``\ earest 
``N``\ eighbor Algorithm\'.

.. end_intro

.. begin_background

Why N2 Was Made
------------------------------------------------------------------------------

Before N2, there has been other great approximate nearest neighbor
libraries such as `Annoy`_ and `NMSLIB`_. However, each of them had
different strengths and weaknesses regarding usability, performance,
and etc. So, N2 has been developed aiming to bring the strengths of
existing aKNN libraries and supplement their weaknesses.

.. end_background

.. begin_features

Features
------------------------------------------------------------------------------

- Lightweight library which runs fast with large datasets.
- Good performance in terms of index build time, search speed,
  and memory usage.
- Supports multi-core CPUs for index building.
- Supports a mmap feature by default to efficiently process large
  index files.
- Supports Python/Go bindings.

.. end_features

Supported Distance Metrics
------------------------------------------------------------------------------

.. Please manually sync the table below with that of docs/index.rst.

+-----------+-------------+--------------------------------------------------------------------+
| Metric    | Definition  | d(**p**, **q**)                                                    |
+-----------+-------------+--------------------------------------------------------------------+
| "angular" | 1 - cosθ    | 1 - {sum(p :sub:`i` · q :sub:`i`) /                                |
|           |             | sqrt(sum(p :sub:`i` · p :sub:`i`) · sum(q :sub:`i` · q :sub:`i`))} |
+-----------+-------------+--------------------------------------------------------------------+
| "L2"      | squared L2  | sum{(p :sub:`i` - q :sub:`i`) :sup:`2`}                            |
+-----------+-------------+--------------------------------------------------------------------+
| "dot"     | dot product | sum(p :sub:`i` · q :sub:`i`)                                       |
+-----------+-------------+--------------------------------------------------------------------+

.. begin_metric_detail

N2 supports three distance metrics.
For "angular" and "L2", **d** (distance) is defined such that the closer the vectors are,
the smaller **d** is. However for "dot", **d** is defined such that the closer
the vectors are, the larger **d** is. You may be wondering why we defined
and implemented "dot" metric as *plain dot product* and not as *(1 - dot product)*.
The rationale for this decision was to allow users to directly interpret the **d** value
returned from Hnsw search function as a dot product value.

.. end_metric_detail

Quickstart
------------------------------------------------------------------------------

1. Install N2 with pip.

.. code:: bash

   $ pip install n2

2. Here is a python code snippet demonstrating how to use N2.

.. code:: python

    import numpy as np

    from n2 import HnswIndex

    N, dim = 10240, 20
    samples = np.arange(N * dim).reshape(N, dim)

    index = HnswIndex(dim)
    for sample in samples:
        index.add_data(sample)
    index.build(m=5, n_threads=4)
    print(index.search_by_id(0, 10))
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Full Documentation
------------------------------------------------------------------------------

Visit `n2.readthedocs.io`_ for full documentation.
The documentation site explains the following contents in detail.

-  `Installation Guide`_

   - how to build from source, etc.

-  User Guide with Basic Examples

   - `Python Interface`_
   - `C++ Interface`_
   - `Go Interface`_

-  `Benchmark`_

   - detailed explanation of how we performed benchmark experiemnts.


Performance
------------------------------------------------------------------------------

- Here are the results of our benchmark experiments.
- You can also see benchmarks of various ANN libraries in Python at `ann-benchmarks.com`_.
  Note that N2 version 0.1.6 is used in `ann-benchmarks.com`_ (last checked on October 8th, 2020)
  and we are continuing our efforts to improve N2 performance.


Index Build Time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|image0|

Search Speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|image1|

Memory Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|image2|


.. begin_footnote

References
------------------------------------------------------------------------------

- Y\. A. Malkov and D. A. Yashunin, "Efficient and robust approximate 
  nearest neighbor search using hierarchical navigable small world 
  graphs," CoRR, vol. abs/1603.09320, 2016. [Online]. 
  Available: http://arxiv.org/abs/1603.09320
-  NMSLIB: https://github.com/nmslib/nmslib
-  Annoy: https://github.com/spotify/annoy

License
------------------------------------------------------------------------------

This software is licensed under the `Apache 2 license`_, quoted below.

Copyright 2017 Kakao Corp. http://www.kakaocorp.com

Licensed under the Apache License, Version 2.0 (the “License”); you may
not use this project except in compliance with the License. You may
obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an “AS IS” BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

.. _Apache 2 license: https://github.com/kakao/n2/blob/master/LICENSE
.. _Annoy: https://github.com/spotify/annoy
.. _NMSLIB: https://github.com/nmslib/nmslib
.. _Installation Guide: https://n2.readthedocs.io/en/latest/install.html
.. _Python Interface: https://n2.readthedocs.io/en/latest/python_api.html
.. _C++ Interface: https://n2.readthedocs.io/en/latest/cpp_api.html
.. _Go Interface: https://n2.readthedocs.io/en/latest/go_api.html
.. _Benchmark: https://n2.readthedocs.io/en/latest/benchmark.html
.. _n2.readthedocs.io: https://n2.readthedocs.io/en/latest/
.. _ann-benchmarks.com: http://ann-benchmarks.com/

.. |image0| image:: docs/imgs/build_time/build_time_threads.png
.. |image1| image:: docs/imgs/search_time/search_time.png
.. |image2| image:: docs/imgs/mem/memory_usage.png

.. end_footnote
