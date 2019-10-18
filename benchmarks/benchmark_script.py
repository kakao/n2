import os
import sys
import abc
import time
import random
import logging
import argparse
import resource
import multiprocessing

import h5py
import numpy
import nmslib

from config import CACHE_DIR, RESULT_DIR, DATA_FILES
from n2 import HnswIndex

try:
    xrange
except NameError:
    xrange = range


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


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric, precision=numpy.float32):
        self._metric = metric
        self._precision = precision

    def fit(self, X):
        """Initialize the search index."""
        lens = (X ** 2).sum(-1)  # precompute (squared) length of each vector
        if self._metric == 'angular':
            X /= numpy.sqrt(lens)[..., numpy.newaxis]  # normalize index vectors to unit length
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)

    def query(self, v, n):
        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        v = numpy.ascontiguousarray(v, dtype=self._precision)  # use same precision for query as for index
        # HACK we ignore query length as that's a constant not affecting the final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * numpy.dot(self.index, v)

        indices = numpy.argpartition(dists, n)[:n]  # partition-sort by distance, get `n` closest
        return sorted(indices, key=lambda index: dists[index])  # sort `n` closest into correct order

    def __str__(self):
        return 'BruteForceBLAS'


class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)
        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(CACHE_DIR, "index_n2_%s_M%d_efCon%d_n_thread%s_data_size%d"
                                        % (args.dataset, m, ef_construction, n_threads, args.data_size))
        self._metric = metric

    def fit(self, X):
        if self._metric == 'euclidean':
            self._n2 = HnswIndex(X.shape[1], 'L2')
        else:
            self._n2 = HnswIndex(X.shape[1])

        if os.path.exists(self._index_name):
            n2_logger.info("Loading index from file")
            self._n2.load(self._index_name)
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
        self._index_name = os.path.join(CACHE_DIR, "index_nmslib_%s_M%d_efCon%d_n_thread%s_data_size%d"
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
    X_train, X_test, corrects = get_dataset(args)
    pool.close()
    pool.join()

    t0 = time.time()
    algo.fit(X_train)
    build_time = time.time() - t0
    n2_logger.debug('Built index in {0}'.format(build_time))

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

    output = '\t'.join(map(str, [library, algo.name, build_time, best_search_time, best_precision]))
    with open(results_fn, 'a') as f:
        f.write(output + '\n')

    n2_logger.info('Summary: {0}'.format(output))


def get_dataset(args):
    cache_fn = get_fn('dataset', args) + '.hdf5'
    if not os.path.exists(cache_fn):
        local_fn = DATA_FILES[args.dataset]
        X = []
        for line in open(local_fn):
            v = [float(x) for x in line.strip().split()]
            X.append(numpy.array(v))
            if len(X) == args.data_size:
                break

        from sklearn.model_selection import train_test_split
        X_train, X_test = train_test_split(X, test_size=args.test_size, random_state=args.random_state)
        write_output(numpy.array(X_train), numpy.array(X_test), cache_fn, args.distance, count=GT_SIZE)

    return load_dataset(cache_fn)


def load_dataset(fn):
    f = h5py.File(fn)
    X_train = numpy.array(f['train'])
    X_test = numpy.array(f['test'])
    corrects = f['neighbors']
    return X_train, X_test, corrects


def write_output(train, test, fn, distance, point_type='float', count=100):
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    f.create_dataset('train', (len(train), len(train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    bf = BruteForceBLAS(distance, precision=train.dtype)
    bf.fit(train)
    for i, x in enumerate(test):
        sys.stderr.write('computing queries %d/%d ...\r' % (i+1, len(test)))
        neighbors[i] = bf.query(x, count)

    sys.stderr.write('\n')
    f.close()


def get_fn(file_type, args, base=CACHE_DIR):
    fn = '%s-%s-%s-%d-%d-%d' % (os.path.join(base, file_type), args.dataset, args.distance,
                                args.data_size, args.test_size, args.random_state)
    return fn


def run(args):
    results_fn = get_fn('result', args, base=RESULT_DIR) + '.txt'

    index_params = [(12, 100)]
    query_params = [25, 50, 100, 250, 500, 750, 1000, 1500, 2500, 5000, 10000]

    algos = {
        'n2': [N2(M, ef_con, args.n_threads, ef_search, 'angular')
               for M, ef_con in index_params
               for ef_search in query_params],
        'nmslib': [NmslibHNSW(M, ef_con, args.n_threads, ef_search, 'angular')
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
    parser.add_argument('--dataset', help='Which dataset',  default='glove', choices=['glove', 'sift', 'youtube'])
    parser.add_argument('--data_size', help='Maximum # of data points (0: unlimited)', type=int, default=0)
    parser.add_argument('--test_size', help='Maximum # of data queries', type=int, default=10000)
    parser.add_argument('--n_threads', help='Number of threads', type=int, default=10)
    parser.add_argument('--random_state', help='Random seed', type=int, default=3)
    parser.add_argument('--algo', help='Algorithm', type=str, choices=['n2', 'nmslib'])
    parser.add_argument('--verbose', '--v', help='print verbose log', type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(DATA_FILES[args.dataset]):
        raise IOError('Please download the dataset')

    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    if args.verbose:
        n2_logger.setLevel(logging.DEBUG)

    numpy.random.seed(args.random_state)

    global GT_SIZE
    GT_SIZE = {'glove': 10, 'sift': 10, 'youtube': 100}[args.dataset]
    n2_logger.debug('GT size: {}'.format(GT_SIZE))

    run(args)
