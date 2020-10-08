.. N2 documentation master file, created by
   sphinx-quickstart on Fri Aug 28 08:38:31 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

N2: Approximate Nearest Neighbor
==============================================================================

.. include:: ../README.rst
   :start-after: begin_intro
   :end-before: end_intro

.. include:: ../README.rst
   :start-after: begin_background
   :end-before: end_background

.. include:: ../README.rst
   :start-after: begin_features
   :end-before: end_features

Supported Distance Metrics
------------------------------------------------------------------------------

.. Please manually sync the table below with that of README.rst.

+-----------+--------------------------+------------------------------------------------------------------+
| Metric    | Definition               | :math:`d(\vec{p}, \vec{q})`                                      |
+-----------+--------------------------+------------------------------------------------------------------+
| "angular" | :math:`1 - \cos{\theta}` | :math:`1 - \frac{\vec{p}\cdot\vec{q}} {||\vec{p}||\ ||\vec{q}||} |
|           |                          | = 1 - \frac{\sum_{i} {p_i q_i}}                                  |
|           |                          | {\sqrt{\sum_{i} {p_i^2}} \sqrt{\sum_{i} {q_i^2}}}`               |
+-----------+--------------------------+------------------------------------------------------------------+
| "L2"      | squared L2               | :math:`\sum_{i} {(p_i - q_i)^2}`                                 |
+-----------+--------------------------+------------------------------------------------------------------+
| "dot"     | dot product              | :math:`\vec{p}\cdot\vec{q} = \sum_{i} {p_i q_i}`                 |
+-----------+--------------------------+------------------------------------------------------------------+

.. include:: ../README.rst
   :start-after: begin_metric_detail
   :end-before: end_metric_detail

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Installation <install>

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   Python Interface <python_api>
   C++ Interface <cpp_api>
   Go Interface <go_api>

.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   Benchmark <benchmark>

.. include:: ../README.rst
   :start-after: begin_footnote
   :end-before: end_footnote
   
.. _Annoy: https://github.com/spotify/annoy
.. _NMSLIB: https://github.com/nmslib/nmslib
