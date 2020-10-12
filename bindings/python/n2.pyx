# cython: experimental_cpp_class_def=True
# -*- coding: utf-8 -*-

# Copyright 2017 Kakao Corp. <http://www.kakaocorp.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict

from libcpp cimport bool as bool_t
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "n2/hnsw.h" namespace "n2":
    cdef cppclass Hnsw:
        Hnsw(int, string) except +
        void SetConfigs(const vector[pair[string, string]]& configs) nogil except +
        bool_t SaveModel(const string&) nogil except +
        bool_t LoadModel(const string&, const bool_t) nogil except +
        void UnloadModel() nogil except +
        void AddData(const vector[float]&) nogil except +
        void Fit() nogil except +
        void SearchByVector(const vector[float]&, size_t, size_t, vector[int]&) nogil except +
        void SearchByVector(const vector[float]&, size_t, size_t, vector[pair[int, float]]&) nogil except +
        void SearchById(int, size_t, size_t, vector[int]&) nogil except +
        void SearchById(int, size_t, size_t, vector[pair[int, float]]&) nogil except +
        void BatchSearchByVectors(const vector[vector[float]]&, size_t, size_t, size_t,
                                  vector[vector[int]]&) nogil except +
        void BatchSearchByVectors(const vector[vector[float]]&, size_t, size_t, size_t,
                                  vector[vector[pair[int, float]]]&) nogil except +
        void BatchSearchByIds(const vector[int]&, size_t, size_t, size_t,
                              vector[vector[int]]&) nogil except +
        void BatchSearchByIds(const vector[int]&, size_t, size_t, size_t,
                              vector[vector[pair[int, float]]]&) nogil except +
        void PrintDegreeDist() nogil except +
        void PrintConfigs() nogil except +

cdef class _HnswIndex:
    cdef Hnsw* obj

    def __cinit__(self, _dim, _metric):
        cdef int dim = _dim
        cdef string metric = _metric.encode('ascii')
        self.obj = new Hnsw(dim, metric)

    def __dealloc__(self):
        del self.obj

    def add_data(self, _v):
        cdef vector[float] v = _v
        with nogil:
            self.obj.AddData(v)

    def save(self, _fname):
        cdef string fname = _fname.encode('ascii')
        with nogil:
            self.obj.SaveModel(fname)

    def load(self, _fname, _use_mmap):
        cdef string fname = _fname.encode('ascii')
        cdef bool_t use_mmap = _use_mmap
        with nogil:
            self.obj.LoadModel(fname, use_mmap)

    def unload(self):
        with nogil:
            self.obj.UnloadModel()

    def build(self, _configs):
        cdef vector[pair[string, string]] configs = _configs
        with nogil:
            self.obj.SetConfigs(configs)
            self.obj.Fit()

    def search_by_vector_incl_dist(self, _v, _k, _ef_search):
        cdef vector[float] v = _v
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef vector[pair[int, float]] ret
        with nogil:
            self.obj.SearchByVector(v, k, ef_search, ret)
        return ret

    def search_by_vector(self, _v, _k, _ef_search):
        cdef vector[float] v = _v
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef vector[int] ret
        with nogil:
            self.obj.SearchByVector(v, k, ef_search, ret)
        return ret

    def search_by_id_incl_dist(self, _item_id, _k, _ef_search):
        cdef int item_id = _item_id
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef vector[pair[int, float]] ret
        with nogil:
            self.obj.SearchById(item_id, k, ef_search, ret)
        return ret

    def search_by_id(self, _item_id, _k, _ef_search):
        cdef int item_id = _item_id
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef vector[int] ret
        with nogil:
            self.obj.SearchById(item_id, k, ef_search, ret)
        return ret

    def batch_search_by_vectors_incl_dist(self, _vs, _k, _ef_search, _num_threads):
        cdef vector[vector[float]] vs = _vs
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef int num_threads = _num_threads
        cdef vector[vector[pair[int, float]]] rets
        with nogil:
            self.obj.BatchSearchByVectors(vs, k, ef_search, num_threads, rets)
        return rets

    def batch_search_by_vectors(self, _vs, _k, _ef_search, _num_threads):
        cdef vector[vector[float]] vs = _vs
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef int num_threads = _num_threads
        cdef vector[vector[int]] rets
        with nogil:
            self.obj.BatchSearchByVectors(vs, k, ef_search, num_threads, rets)
        return rets

    def batch_search_by_ids_incl_dist(self, _item_ids, _k, _ef_search, _num_threads):
        cdef vector[int] item_ids = _item_ids
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef int num_threads = _num_threads
        cdef vector[vector[pair[int, float]]] rets
        with nogil:
            self.obj.BatchSearchByIds(item_ids, k, ef_search, num_threads, rets)
        return rets

    def batch_search_by_ids(self, _item_ids, _k, _ef_search, _num_threads):
        cdef vector[int] item_ids = _item_ids
        cdef size_t k = _k
        cdef size_t ef_search = _ef_search
        cdef int num_threads = _num_threads
        cdef vector[vector[int]] rets
        with nogil:
            self.obj.BatchSearchByIds(item_ids, k, ef_search, num_threads, rets)
        return rets

    def print_degree_dist(self):
        with nogil:
            self.obj.PrintDegreeDist()

    def print_configs(self):
        with nogil:
            self.obj.PrintConfigs()


class HnswIndex(object):
    def __init__(self, dimension, metric='angular'):
        """

        Args:
            dimension (int): Dimension of vectors.
            metric (string): An optional parameter to choose a distance metric.
                            ('angular' | 'L2' | 'dot')

        Returns:
            An instance of Hnsw index.

        """
        self.model = _HnswIndex(dimension, metric)

    def add_data(self, v):
        """Adds vector v.

        Args:
            v (list(float)): A vector with dimension ``dimension`` set in __init__().

        Returns:
            bool: Boolean value indicating whether data addition succeeded or not.

        """
        return self.model.add_data(v)

    def save(self, fname):
        """Saves the index to disk.

        Args:
            fname (str): A file destination where the index will be saved.

        Returns:
            bool: Boolean value indicating whether model save succeeded or not.

        """
        return self.model.save(fname)

    def load(self, fname, use_mmap=True):
        """Loads the index from disk.

        Args:
            fname (str): An index file name.
            use_mmap (bool): An optional parameter indicating whether to use
                mmap() or not (default: True).
                If this parameter is set, N2 loads model through mmap.

        Returns:
            bool: Boolean value indicating whether model load succeeded or not.

        """
        return self.model.load(fname, use_mmap)

    def unload(self):
        """Unloads (unmap) the index.
        """
        self.model.unload()

    def build(self, m=None, max_m0=None, ef_construction=None, n_threads=None,
              mult=None, neighbor_selecting=None, graph_merging=None):
        """Builds a hnsw graph with given configurations.

        Args:
            m (int): Max number of edges for nodes at level > 0 (default: 12).
            max_m0 (int): Max number of edges for nodes at level == 0 (default: 24).
            ef_construction (int): Refer to HNSW paper (default: 150).
            n_threads (int): Number of threads for building index.
            mult (float): Level multiplier. Recommended to use the default value
                (default: 1 / log(1.0 * M)).
            neighbor_selecting (string): Neighbor selecting policy.

                - Available values
                    -  ``"heuristic"`` (default): Select neighbors using algorithm4 on HNSW paper (recommended).
                    -  ``"naive"``: Select closest neighbors (not recommended).

            graph_merging (string): Graph merging heuristic.

                - Available values
                    -  ``"skip"`` (default): Do not merge (recommended for large-scale data (over 10M)).
                    -  ``"merge_level0"``: Performs an additional graph build in reverse order,
                       then merges edges at level 0. So, it takes twice the build time compared to
                       ``"skip"`` but shows slightly higher accuracy. (recommended for data under 10M scale).

        """
        configs = []
        if m is not None:
            configs.append(['M'.encode('ascii'), str(m).encode('ascii')])
        if max_m0 is not None:
            configs.append(['MaxM0'.encode('ascii'), str(max_m0).encode('ascii')])
        if ef_construction is not None:
            configs.append(['efConstruction'.encode('ascii'), str(ef_construction).encode('ascii')])
        if n_threads is not None:
            configs.append(['NumThread'.encode('ascii'), str(n_threads).encode('ascii')])
        if mult is not None:
            configs.append(['Mult'.encode('ascii'), str(mult).encode('ascii')])
        if neighbor_selecting is not None:
            configs.append(['NeighborSelecting'.encode('ascii'), neighbor_selecting.encode('ascii')])
        if graph_merging is not None:
            configs.append(['GraphMerging'.encode('ascii'), graph_merging.encode('ascii')])
        return self.model.build(configs)

    def search_by_vector(self, v, k, ef_search=-1, include_distances=False):
        """Returns k nearest items (as vectors) to a query item.

        Args:
            v (list(float)): A query vector.
            k (int): k value.
            ef_search (int): ef_search metric (default: 50 * k).
                If you pass -1 to ef_search, ef_search will be set as the default value.
            include_distances (bool): If you set this argument to True,
                it will return a list of tuples((item_id, distance)).

        Returns:
            list(int) or list(tuple(int, float)): A list of k nearest items.

        """
        if ef_search == -1:
            ef_search = k * 50
        if include_distances:
            return self.model.search_by_vector_incl_dist(v, k, ef_search)
        else:
            return self.model.search_by_vector(v, k, ef_search)

    def search_by_id(self, item_id, k, ef_search=-1, include_distances=False):
        """Returns k nearest items (as ids) to a query item.

        Args:
            item_id (int): A query id.
            k (int): k value.
            ef_search (int): ef_search metric (default: 50 * k).
                If you pass -1 to ef_search, ef_search will be set as the default value.
            include_distances (bool): If you set this argument to True,
                it will return a list of tuples((item_id, distance)).

        Returns:
            list(int) or list(tuple(int, float)): A list of k nearest items.

        """
        if ef_search == -1:
            ef_search = k * 50
        if include_distances:
            return self.model.search_by_id_incl_dist(item_id, k, ef_search)
        else:
            return self.model.search_by_id(item_id, k, ef_search)

    def batch_search_by_vectors(self, vs, k, ef_search=-1, num_threads=4, include_distances=False):
        """Returns k nearest items (as vectors) to each query item (batch search with multi-threads).

        Note:
            With OMP_SCHEDULE environment variable, you can set how threads are scheduled.
            Refer to `GNU libgomp <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fSCHEDULE.html#OMP_005fSCHEDULE>`_.

        Args:
            vs (list(list(float))): Query vectors.
            k (int): k value.
            ef_search (int): ef_search metric (default: 50 * k).
                If you pass -1 to ef_search, ef_search will be set as the default value.
            num_threads (int): Number of threads to use for search.
            include_distances (bool): If you set this argument to True,
                it will return a list of tuples((item_id, distance)).

        Returns:
            list(list(int) or list(list(tuple(int, float))): A list of list of
            k nearest items for each input query item in the order passed to parameter ``vs``.

        """
        if ef_search == -1:
            ef_search = k * 50
        if include_distances:
            return self.model.batch_search_by_vectors_incl_dist(vs, k, ef_search, num_threads)
        else:
            return self.model.batch_search_by_vectors(vs, k, ef_search, num_threads)

    def batch_search_by_ids(self, item_ids, k, ef_search=-1, num_threads=4, include_distances=False):
        """Returns k nearest items (as ids) to each query item (batch search with multi-threads).

        Note:
            With OMP_SCHEDULE environment variable, you can set how threads are scheduled.
            Refer to `GNU libgomp <https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fSCHEDULE.html#OMP_005fSCHEDULE>`_.

        Args:
            item_ids (list(int)): Query ids.
            k (int): k value.
            ef_search (int): ef_search metric (default: 50 * k).
                If you pass -1 to ef_search, ef_search will be set as the default value.
            num_threads (int): Number of threads to use for search.
            include_distances (bool): If you set this argument to True,
                it will return a list of tuples((item_id, distance)).

        Returns:
            list(list(int) or list(list(tuple(int, float))): A list of list of
            k nearest items for each input query item in the order passed to parameter ``item_ids``.

        """
        if ef_search == -1:
            ef_search = k * 50
        if include_distances:
            return self.model.batch_search_by_ids_incl_dist(item_ids, k, ef_search, num_threads)
        else:
            return self.model.batch_search_by_ids(item_ids, k, ef_search, num_threads)

    def print_degree_dist(self):
        """Prints degree distributions.
        """
        self.model.print_degree_dist()

    def print_configs(self):
        """Prints configurations.
        """
        self.model.print_configs()
