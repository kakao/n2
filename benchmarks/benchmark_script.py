import os
import sys
import gzip
import time
import pickle
import random
import logging
import argparse
import resource
import multiprocessing

import numpy
import nmslib

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


INDEX_DIR = './indices/'
DATA_DIR = './datasets/'
DATA_FILES = {
    'glove': DATA_DIR + 'glove.txt',
    'sift': DATA_DIR + 'sift.txt',
    'youtube': DATA_DIR + 'youtube.txt'}


class BaseANN(object):
    name = None

    def use_threads(self):
        return True

    def __str__(self):
        return self.name


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric, precision=numpy.float32):
        self._metric = metric
        self._precision = precision
        self.name = 'BruteForceBLAS()'

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


class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)

        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(INDEX_DIR, "n2_%s_M%d_efCon%d_n_thread%s_data_size%d"
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
        else:
            n2_logger.debug("Index file is not exist: {0}".format(self._index_name))
            n2_logger.info("Start fitting")

            for i, x in enumerate(X):
                self._n2.add_data(x.tolist())
            self._n2.build(m=self._m, max_m0=self._m0, ef_construction=self._ef_construction, n_threads=self._n_threads)
            self._n2.save(self._index_name)

    def query(self, v, n):
        return self._n2.search_by_vector(v.tolist(), n, self._ef_search)


class NmslibHNSW(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self.name = "nmslib_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)

        self._index_param = [
            'M=%d' % m,
            'indexThreadQty=%d' % n_threads,
            'efConstruction=%d' % ef_construction,
            'post=0', 'delaunay_type=2']
        self._query_param = ['efSearch=%d' % ef_search]
        self._index_name = os.path.join(INDEX_DIR, "nmslib_%s_M%d_efCon%d_n_thread%s_data_size%d"
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
                self._index.addDataPoint(i, x.tolist())

            nmslib.createIndex(self._index, self._index_param)
            nmslib.saveIndex(self._index, self._index_name)

        nmslib.setQueryTimeParams(self._index, self._query_param)

    def query(self, v, n):
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        nmslib.freeIndex(self._index)


def run_algo(args, library, algo, results_fn):
    pool = multiprocessing.Pool()
    X_train, X_test = get_dataset(which=args.dataset, data_size=args.data_size,
                                  test_size=args.test_size, random_state=args.random_state)
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
        total_queries = len(queries)
        for j in range(total_queries):
            sys.stderr.write("[%d/%d][algo: %s] Querying: %d / %d \r"
                             % (i+1, try_count, str(algo), j+1, total_queries))
            v, correct = queries[j]
            t0 = time.time()
            found = algo.query(v, GT_SIZE)
            search_time += (time.time() - t0)
            if len(found) < len(correct):
                n2_logger.debug('found: {0}, correct: {1}'.format(len(found), len(correct)))
            results.append(len(set(found).intersection(correct)))
        sys.stderr.write("\n")

        k = float(sum(results))
        search_time /= len(queries)
        precision = k / (len(queries) * GT_SIZE)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)
        n2_logger.debug('[%d/%d][algo: %s] search time: %s, precision: %.5f'
                        % (i+1, try_count, str(algo), str(search_time), precision))

    output = '\t'.join(map(str, [library, algo.name, build_time, best_search_time, best_precision]))
    with open(results_fn, 'a') as f:
        f.write(output + '\n')
    n2_logger.info('Summary: {0}'.format(output))


def get_dataset(which='glove', data_size=0, test_size=10000, random_state=3):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, data_size, test_size, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        n2_logger.debug('{0} {1}'.format(X_train.shape, X_test.shape))
        return X_train, X_test

    local_fn = os.path.join('datasets', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    X = []
    for line in f:
        v = [float(x) for x in line.strip().split()]
        X.append(v)
        if len(X) == data_size:
            break

    X = numpy.vstack(X)
    import sklearn.model_selection

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = sklearn.model_selection.train_test_split(X, test_size=test_size, random_state=random_state)
    X_train = X_train.astype(numpy.float)
    X_test = X_test.astype(numpy.float)
    n2_logger.debug('{0} {1}'.format(X_train.shape, X_test.shape))
    numpy.savez(cache, train=X_train, test=X_test)
    return X_train, X_test


def get_queries(args):
    n2_logger.debug('computing queries with correct results...')

    X_train, X_test = get_dataset(which=args.dataset, data_size=args.data_size,
                                  test_size=args.test_size, random_state=args.random_state)

    bf = BruteForceBLAS(args.distance)
    bf.fit(X_train)

    queries = []
    total_queries = len(X_test)
    for x in X_test:
        correct = bf.query(x, GT_SIZE)
        queries.append((x, correct))
        sys.stderr.write('computing queries %d/%d ...\r' % (len(queries), total_queries))
    sys.stderr.write('\n')
    return queries


def get_fn(base, args):
    fn = '%s-%d-%d-%d.txt' % (os.path.join(base, args.dataset), args.data_size, args.test_size, args.random_state)
    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)
    return fn


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

    if args.verbose:
        n2_logger.setLevel(logging.DEBUG)

    numpy.random.seed(args.random_state)

    global GT_SIZE
    GT_SIZE = {'glove': 10, 'sift': 10, 'youtube': 100}[args.dataset]
    n2_logger.debug('GT size: {}'.format(GT_SIZE))

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)
    logging.info('storing queries in {0} and results in {1}.'.format(queries_fn, results_fn))

    if not os.path.exists(queries_fn):
        queries = get_queries(args)
        with open(queries_fn, 'wb') as f:
            pickle.dump(queries, f)
    else:
        queries = pickle.load(open(queries_fn, 'rb'))

    logging.debug('got {0} queries'.format(len(queries)))

    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)

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
