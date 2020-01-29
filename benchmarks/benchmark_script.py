import os
import sys
import abc
import time
import random
import psutil
import logging
import argparse
import resource
import multiprocessing

import h5py
import numpy
import nmslib

from download_dataset import get_dataset_fn, DATASETS
from n2 import HnswIndex

try:
    xrange
except NameError:
    xrange = range


CACHE_DIR = './cache'
RESULT_DIR = './result'
GT_SIZE = 100


logging.basicConfig(format='%(message)s')
n2_logger = logging.getLogger("n2_benchmark")
n2_logger.setLevel(logging.INFO)


# Set resource limits to prevent memory bombs
memory_limit = 64 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    n2_logger.debug('resetting memory limit from {0} to {1}. '.format(soft, memory_limit))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))


class BaseANN(object):
    @abc.abstractmethod
    def fit(self, X):
        pass

    @abc.abstractmethod
    def query(self, v, n):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024


class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)
        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(CACHE_DIR, "index_n2_%s_M%d_efCon%d_n_thread%s_datasz%d"
                                        % (args.dataset, m, ef_construction, n_threads, args.data_size))
        self._metric = metric

    def fit(self, X):
        if self._metric == 'euclidean':
            self._n2 = HnswIndex(X.shape[1], 'L2')
        else:
            self._n2 = HnswIndex(X.shape[1])

        if os.path.exists(self._index_name):
            n2_logger.info("Loading index from file")
            self._n2.load(self._index_name, use_mmap=False)
            return

        n2_logger.debug("Create Index")
        for i, x in enumerate(X):
            self._n2.add_data(x)
        self._n2.build(m=self._m, max_m0=self._m0, ef_construction=self._ef_construction, n_threads=self._n_threads)
        self._n2.save(self._index_name)

    def query(self, v, n):
        return self._n2.search_by_vector(v, n, self._ef_search)

    def __str__(self):
        return self.name


class NmslibHNSW(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self.name = "nmslib_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)
        self._index_param = [
            'M=%d' % m,
            'indexThreadQty=%d' % n_threads,
            'efConstruction=%d' % ef_construction,
            'post=0', 'delaunay_type=2']
        self._query_param = ['efSearch=%d' % ef_search]
        self._index_name = os.path.join(CACHE_DIR, "index_nmslib_%s_M%d_efCon%d_n_thread%s_datasz%d"
                                        % (args.dataset, m, ef_construction, n_threads, args.data_size))
        self._metric = {'angular': 'cosinesimil', 'euclidean': 'l2'}[metric]

    def fit(self, X):
        self._index = nmslib.init(self._metric, [], "hnsw", nmslib.DataType.DENSE_VECTOR, nmslib.DistType.FLOAT)

        if os.path.exists(self._index_name):
            logging.debug("Loading index from file")
            nmslib.loadIndex(self._index, self._index_name)
        else:
            logging.debug("Create Index")
            for i, x in enumerate(X):
                self._index.addDataPoint(i, x)

            nmslib.createIndex(self._index, self._index_param)
            nmslib.saveIndex(self._index, self._index_name)

        nmslib.setQueryTimeParams(self._index, self._query_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v)

    def free_index(self):
        nmslib.freeIndex(self._index)

    def __str__(self):
        return self.name


def run_algo(args, library, algo, results_fn):
    pool = multiprocessing.Pool()
    X_train, X_test, corrects = load_dataset(args.dataset)
    pool.close()
    pool.join()

    memory_usage_before = algo.get_memory_usage()
    t0 = time.time()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size_kb = algo.get_memory_usage() - memory_usage_before
    n2_logger.info('Built index in {0}, Index size: {1}KB'.format(build_time, index_size_kb))

    best_search_time = float('inf')
    best_precision = 0.0  # should be deterministic but paranoid
    try_count = args.try_count
    for i in xrange(try_count):  # Do multiple times to warm up page cache, use fastest
        results = []
        search_time = 0.0
        for j, v in enumerate(X_test):
            sys.stderr.write("[%d/%d][algo: %s] Querying: %d / %d \r"
                             % (i+1, try_count, str(algo), j+1, len(X_test)))
            t0 = time.time()
            found = algo.query(v, GT_SIZE)
            search_time += (time.time() - t0)

            results.append(len(set(found).intersection(corrects[j])))

            if len(found) < len(corrects[j]):
                n2_logger.debug('found: {0}, correct: {1}'.format(len(found), len(corrects[j])))

        sys.stderr.write("\n")

        k = float(sum(results))
        search_time /= len(X_test)
        precision = k / (len(X_test) * GT_SIZE)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)
        n2_logger.debug('[%d/%d][algo: %s] search time: %s, precision: %.5f'
                        % (i+1, try_count, str(algo), str(search_time), precision))

    output = '\t'.join(map(str, [library, algo.name, build_time, best_search_time, best_precision, index_size_kb]))
    with open(results_fn, 'a') as f:
        f.write(output + '\n')

    n2_logger.info('Summary: {0}'.format(output))


def load_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    f = h5py.File(hdf5_fn, 'r')
    X_train = numpy.array(f['train'])
    X_test = numpy.array(f['test'])
    corrects = f['neighbors']
    return X_train, X_test, corrects


def get_fn(file_type, args, base=CACHE_DIR):
    fn = '%s_%s_%d_%d_%d' % (os.path.join(base, file_type), args.dataset,
                             args.data_size, args.test_size, args.random_state)
    return fn


def run(args):
    results_fn = get_fn('result', args, base=RESULT_DIR) + '.txt'

    index_params = [(12, 100)]
    query_params = [25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 10000]

    algos = {
        'n2': [N2(M, ef_con, args.n_threads, ef_search, args.distance)
               for M, ef_con in index_params
               for ef_search in query_params],
        'nmslib': [NmslibHNSW(M, ef_con, args.n_threads, ef_search, args.distance)
                   for M, ef_con in index_params
                   for ef_search in query_params],
    }

    if args.algo:
        algos = {args.algo: algos[args.algo]}

    algos_flat = [(k, v) for k, vals in algos.items() for v in vals]
    random.shuffle(algos_flat)
    n2_logger.debug('order: %s' % str([a.name for l, a in algos_flat]))

    for library, algo in algos_flat:
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(target=run_algo, args=(args, library, algo, results_fn))
        p.start()
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', help='Distance metric', default='angular', choices=['angular', 'euclidean'])
    parser.add_argument('--try_count', help='Number of test attempts', type=int, default=3)
    parser.add_argument('--dataset', help='Which dataset',  default='glove', choices=DATASETS)
    parser.add_argument('--data_size', help='Maximum # of data points (0: unlimited)', type=int, default=0)
    parser.add_argument('--test_size', help='Maximum # of data queries', type=int, default=10000)
    parser.add_argument('--n_threads', help='Number of threads', type=int, default=10)
    parser.add_argument('--random_state', help='Random seed', type=int, default=3)
    parser.add_argument('--algo', help='Algorithm', type=str, choices=['n2', 'nmslib'])
    parser.add_argument('--verbose', '--v', help='print verbose log', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(get_dataset_fn(args.dataset)):
        raise IOError('Please download the dataset')

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if args.verbose:
        n2_logger.setLevel(logging.DEBUG)

    numpy.random.seed(args.random_state)

    run(args)
