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
import nmslib
import numpy as np

from n2 import HnswIndex
from metrics import knn_recall, metrics
from download_dataset import get_dataset_fn, DATASETS


try:
    xrange
except NameError:
    xrange = range


CACHE_DIR = './cache'
RESULT_DIR = './result'


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
    def batch_query(self, X, n):
        pass

    @abc.abstractmethod
    def get_batch_results(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    def get_memory_usage(self):
        return psutil.Process().memory_info().rss / 1024


class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric, batch):
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d%s" % (m, ef_construction, n_threads, ef_search,
                                                                '_batch' if batch else '')
        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(CACHE_DIR, "index_n2_%s_M%d_efCon%d_n_thread%s"
                                        % (args.dataset, m, ef_construction, n_threads))
        self._metric = metric

    def fit(self, X):
        if self._metric == 'euclidean':
            self._n2 = HnswIndex(X.shape[1], 'L2')
        elif self._metric == 'dot':
            self._n2 = HnswIndex(X.shape[1], 'dot')
        else:
            self._n2 = HnswIndex(X.shape[1])

        if os.path.exists(self._index_name):
            n2_logger.info("Loading index from file")
            self._n2.load(self._index_name, use_mmap=False)
            return

        n2_logger.info("Create Index")
        for i, x in enumerate(X):
            self._n2.add_data(x)
        self._n2.build(m=self._m, max_m0=self._m0, ef_construction=self._ef_construction, n_threads=self._n_threads)
        self._n2.save(self._index_name)

    def query(self, v, n):
        return self._n2.search_by_vector(v, n, self._ef_search)

    def batch_query(self, X, n):
        self.b_res = self._n2.batch_search_by_vectors(X, n, self._ef_search, self._n_threads)

    def get_batch_results(self):
        return self.b_res

    def __str__(self):
        return self.name


class NmslibHNSW(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric, batch):
        self.name = "NMSLIB_M%d_efCon%d_n_thread%s_efSearch%d%s" % (m, ef_construction, n_threads, ef_search,
                                                                    '_batch' if batch else '')
        self._index_param = [
            'M=%d' % m,
            'indexThreadQty=%d' % n_threads,
            'efConstruction=%d' % ef_construction,
            'post=0', 'delaunay_type=2']
        self._query_param = ['efSearch=%d' % ef_search]
        self._index_name = os.path.join(CACHE_DIR, "index_nmslib_%s_M%d_efCon%d_n_thread%s"
                                        % (args.dataset, m, ef_construction, n_threads))
        self._n_threads = n_threads
        self._metric = {'angular': 'cosinesimil', 'euclidean': 'l2', 'dot': None}[metric]

    def fit(self, X):
        self._index = nmslib.init(self._metric, [], "hnsw", nmslib.DataType.DENSE_VECTOR, nmslib.DistType.FLOAT)

        if os.path.exists(self._index_name):
            logging.info("Loading index from file")
            self._index.loadIndex(self._index_name)
        else:
            logging.info("Create Index")
            for i, x in enumerate(X):
                self._index.addDataPoint(i, x)

            self._index.createIndex(self._index_param)
            self._index.saveIndex(self._index_name)

        self._index.setQueryTimeParams(self._query_param)

    def query(self, v, n):
        ids, distances = self._index.knnQuery(v, n)
        return ids

    def batch_query(self, X, n):
        self.b_res = self._index.knnQueryBatch(X, n, self._n_threads)

    def get_batch_results(self):
        return [x for x, _ in self.b_res]

    def __str__(self):
        return self.name


def run_algo(args, library, algo, results_fn):
    n2_logger.info('algo: {0}'.format(algo))
    pool = multiprocessing.Pool()
    pool.close()
    pool.join()

    db = load_db(args.dataset)
    X_train = load_train_data(db)

    memory_usage_before = algo.get_memory_usage()
    t0 = time.time()
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size_kb = algo.get_memory_usage() - memory_usage_before
    n2_logger.info('Built index in {0}, Index size: {1}KB'.format(build_time, index_size_kb))

    X_test, nn_dists = load_test_data(db, args.dataset)

    best_search_time = float('inf')
    best_recall = 0.0  # should be deterministic but paranoid

    if not args.build_only:
        try_count = args.try_count
        for i in xrange(try_count):  # Do multiple times to warm up page cache, use fastest
            if args.batch:
                t0 = time.time()
                algo.batch_query(X_test, args.count)
                search_time = (time.time() - t0)
                b_found = algo.get_batch_results()
                b_found_dists = [[float(metrics[args.distance]['distance'](v, X_train[k]))
                                  for k in found]
                                 for v, found in zip(X_test, b_found)]
                recall = sum(knn_recall(dists, found_dists, args.count)
                             for dists, found_dists in zip(nn_dists, b_found_dists))
            else:
                recall = 0.0
                search_time = 0.0
                for j, v in enumerate(X_test):
                    sys.stderr.write("[%d/%d][algo: %s] Querying: %d / %d \r"
                                     % (i+1, try_count, str(algo), j+1, len(X_test)))
                    t0 = time.time()
                    found = algo.query(v, args.count)
                    search_time += (time.time() - t0)
                    found_dists = [float(metrics[args.distance]['distance'](v, X_train[k])) for k in found]
                    recall += knn_recall(nn_dists[j], found_dists, args.count)
                sys.stderr.write("\n")

            search_time /= len(X_test)
            recall /= len(X_test)
            best_search_time = min(best_search_time, search_time)
            best_recall = max(best_recall, recall)
            n2_logger.info('[%d/%d][algo: %s] search time: %s, recall: %.5f'
                           % (i+1, try_count, str(algo), str(search_time), recall))

    db.close()
    output = '\t'.join(map(str, [library, algo.name, build_time, best_search_time, best_recall, index_size_kb]))
    with open(results_fn, 'a') as f:
        f.write(output + '\n')

    n2_logger.info('Summary: {0}\n'.format(output))


def load_db(which):
    hdf5_fn = get_dataset_fn(which)
    db = h5py.File(hdf5_fn, 'r')
    return db


def load_train_data(db):
    return np.array(db['train'])


def load_test_data(db, which):
    test = np.array(db['test'])
    try:
        distances = np.array(db['distances'])
    except KeyError:
        if which in ['youtube1m-40-angular', 'youtube-40-angular']:
            n2_logger.error('Your "%s" dataset may be outdated. Remove it and download again.' % which)
        sys.exit('"distances" does not exists in the hdf5 database.')
    return test, distances


def get_fn(file_type, args, base=CACHE_DIR):
    fn = '%s_%s_%d_%d' % (os.path.join(base, file_type), args.dataset, args.count, args.random_state)
    return fn


def run(args):
    results_fn = get_fn('result', args, base=RESULT_DIR) + '.txt'

    index_params = [(12, 100)]
    query_params = args.ef_searches or [25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000]
    if args.build_only:
        query_params = query_params[:1]

    algos = {
        'n2': [N2(M, ef_con, args.n_threads, ef_search, args.distance, args.batch)
               for M, ef_con in index_params
               for ef_search in query_params],
        'nmslib': [NmslibHNSW(M, ef_con, args.n_threads, ef_search, args.distance, args.batch)
                   for M, ef_con in index_params
                   for ef_search in query_params],
    }

    if args.algo:
        algos = {args.algo: algos[args.algo]}

    algos_flat = [(k, v) for k, vals in algos.items() for v in vals]
    random.shuffle(algos_flat)
    n2_logger.debug('order: %s' % str([a.name for _, a in algos_flat]))

    for library, algo in algos_flat:
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(target=run_algo, args=(args, library, algo, results_fn))
        p.start()
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', help='Distance metric', default='angular', choices=['angular', 'euclidean', 'dot'])
    parser.add_argument('--count', '-k', help="the number of nn to search for", type=int, default=100)
    parser.add_argument('--try_count', help='Number of test attempts', type=int, default=3)
    parser.add_argument('--dataset', help='Which dataset',  default='glove-100-angular', choices=DATASETS)
    parser.add_argument('--n_threads', help='Number of threads', type=int, default=10)
    parser.add_argument('--random_state', help='Random seed', type=int, default=3)
    parser.add_argument('--algo', help='Algorithm', type=str, choices=['n2', 'nmslib'])
    parser.add_argument('--batch', help='Batch search mode with multi-threading', action='store_true')
    parser.add_argument('--build_only', help='Benchmark only build time and memory usage', action='store_true')
    parser.add_argument('--ef_searches', help='Custom ef_search query parameters', type=int, nargs='*')
    parser.add_argument('--verbose', '-v', help='Print verbose log', action='store_true')
    args = parser.parse_args()

    if not os.path.exists(get_dataset_fn(args.dataset)):
        raise IOError('Please download the dataset')

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if args.verbose:
        n2_logger.setLevel(logging.DEBUG)

    random.seed(args.random_state)

    run(args)
