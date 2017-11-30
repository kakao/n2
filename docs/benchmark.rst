N2 Benchmark explanation
========================

Benchmark Focus
---------------

First, the new approximate nearest neighborhoods algorithm should run
faster for large datasets. Second, online content services like news
portal, where dataset is frequently updated(e.g. create/update/delete),
building the index file should be done in minimum time. Therefore, these
are our main criteria:

-  How long does it take to build the index file?
-  How long does it take to get results from the large dataset?
-  How large memory does it take to run large dataset?

Test dataset
------------

Dataset description
~~~~~~~~~~~~~~~~~~~

The dataset consists of two files. ``youtube.txt`` contains 14520986
samples, each sample has 40 data points.

+-------------------+-------------------+----+-------------------+--------------------+
| feature1(float32) | feature2(float32) | …… | feature2(float32) | feature40(float32) |
+===================+===================+====+===================+====================+
|     -0.167898     |     0.160478      | …… |    0.104421       |    0.0503584       |
+-------------------+-------------------+----+-------------------+--------------------+

``youtube.txt.vids`` is a metadata informations of the dataset. Each
line is the metadata corresponding to each sample of ``youtube.txt``.

+------------------+-------------+-------------------------------------------+
|       DSID       |     VID     |              Youtube link                 |
+==================+=============+===========================================+
|34XnPr4YKpo8wE_mEl| Z1Jilm0TZHY | http://www.youtube.com/watch?v=Z1Jilm0TZHY|
+------------------+-------------+-------------------------------------------+

How to get it
~~~~~~~~~~~~~

We share our dataset through `google
drive <https://drive.google.com/open?id=1B3PWRTb8xol9fEkawVbpfitOsuwXkqss>`__.

Index build times
-----------------

|image0|

Generally N2 is faster than the nmslib to build index file. Compared to
the annoy, N2 begins to show the similar performance of the annoy when
N2 uses 5 threads, and from then on it shows a faster build performance
than the annoy.

+-----------+--------------+-------------+-------------+-------------+
| Library   | 1 thread     | 5 threads   | 10 threads  | 20 threads  |
+===========+==============+=============+=============+=============+
| N2        | 4505.3669540 | 1002.475105 | 591.6419599 | 478.1210601 |
| (3.1Gb)   | 9            | 05          | 06          | 33          |
|           | sec          | sec         | sec         | sec         |
+-----------+--------------+-------------+-------------+-------------+
| nmslib    | 7130.7202520 | 1453.570172 | 826.9151070 | 602.1200079 |
| (3.4Gb)   | 4            | 07          | 12          | 92          |
|           | sec          | sec         | sec         | sec         |
+-----------+--------------+-------------+-------------+-------------+
| annoy     | 915.41107487 | 915.4110748 | 915.4110748 | 915.4110748 |
| (4.4Gb)   | 7            | 77          | 77          | 77          |
|           | sec          | sec         | sec         | sec         |
+-----------+--------------+-------------+-------------+-------------+

Search speed
------------

|image1|

+-----------------------------------------+-----------------------+----------+
| Library                                 | Search time           | Accuracy |
+=========================================+=======================+==========+
| Linear search (numpy based)             | 0.358749273825 sec    | 1.0      |
+-----------------------------------------+-----------------------+----------+
| N2 (efCon = 100, efSearch = 10)         | 2.98758983612e-05 sec | 0.054243 |
+-----------------------------------------+-----------------------+----------+
| N2 (efCon = 100, efSearch = 100)        | 0.000128486037254 sec | 0.48313  |
+-----------------------------------------+-----------------------+----------+
| N2 (efCon = 100, efSearch = 1000)       | 0.000824773144722 sec | 0.840634 |
+-----------------------------------------+-----------------------+----------+
| N2 (efCon = 100, efSearch = 10000)      | 0.00720949418545 sec  | 0.926739 |
+-----------------------------------------+-----------------------+----------+
| N2 (efCon = 100, efSearch = 100000)     | 0.0763142487288 sec   | 0.940606 |
+-----------------------------------------+-----------------------+----------+
| Nmslib (efCon = 100, efSearch = 10)     | 9.8201584816e-05 sec  | 0.226192 |
+-----------------------------------------+-----------------------+----------+
| Nmslib (efCon = 100, efSearch = 100)    | 0.000225761222839 sec | 0.672228 |
+-----------------------------------------+-----------------------+----------+
| Nmslib (efCon = 100, efSearch = 1000)   | 0.00140970699787 sec  | 0.882695 |
+-----------------------------------------+-----------------------+----------+
| Nmslib (efCon = 100, efSearch = 10000)  | 0.0143689704418 sec   | 0.935395 |
+-----------------------------------------+-----------------------+----------+
| Nmslib (efCon = 100, efSearch = 100000) | 0.159999159241 sec    | 0.94283  |
+-----------------------------------------+-----------------------+----------+
| Annoy(n_trees=10, search_k=7)           | 4.04834747314e-05 sec | 0.05471  |
+-----------------------------------------+-----------------------+----------+
| Annoy(n_trees=10, search_k=3000)        | 0.00096510682106 sec  | 0.481099 |
+-----------------------------------------+-----------------------+----------+
| Annoy(n_trees=10, search_k=50000)       | 0.0144059297085 sec   | 0.835895 |
+-----------------------------------------+-----------------------+----------+
| Annoy(n_trees=10, search_k=200000)      | 0.053891249156 sec    | 0.918569 |
+-----------------------------------------+-----------------------+----------+
| Annoy(n_trees=10, search_k=500000)      | 0.108285815144 sec    | 0.940851 |
+-----------------------------------------+-----------------------+----------+

Overall, we can see that N2 has a much higher accuracy than the annoy,
and N2 has better performance than the other two libraries at high
precision points.

Memory usage
------------

|image2|

+---------+-------------------+-------------------------------+
| Library | Peak memory usage | Search time peak memory usage |
+=========+===================+===============================+
| N2      | 5360.93750Mb      | 3441.13281Mb                  |
+---------+-------------------+-------------------------------+
| annoy   | 5360.89844Mb      | 3441.09375Mb                  |
+---------+-------------------+-------------------------------+
| nmslib  | 5360.97656Mb      | 3441.17188 Mb                 |
+---------+-------------------+-------------------------------+

The three libraries do not show much difference in memory usage.

Conclusion
----------

In short, on multi-core CPU, N2 performs best. annoy is a good choice
for small datasets that can be handled by a single thread. However, when
dataset is large, where high indexing performance is critical, N2 is
where to go. N2 runs almost 2x faster than annoy. When high precision is
required, both nmslib and N2 are good.

.. |image0| image:: imgs/build_time/build_time_threads.png
.. |image1| image:: imgs/search_time/total.png
.. |image2| image:: imgs/mem/memory_usage.png
