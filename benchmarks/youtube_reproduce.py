import gzip
import numpy
import time
import os
import multiprocessing
import argparse
import pickle
import resource
import random
import math
import logging
import shutil
import subprocess
import sys
import tarfile

from contextlib import closing

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

try:
    xrange
except NameError:
    xrange = range

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

# Set resource limits to prevent memory bombs
memory_limit = 12 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    logging.debug('resetting memory limit from {0} to {1}. '.format(
                  soft, memory_limit))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

INDEX_DIR = 'indices'
DATA_DIR = './datasets/'
YOUTUBE_DIR = DATA_DIR + 'youtube.txt'


class BaseANN(object):
    def use_threads(self):
        return True


class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""

    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError(
                "BruteForceBLAS doesn't support metric %s" %
                metric)
        self._metric = metric
        self._precision = precision
        self.name = 'BruteForceBLAS()'

    def fit(self, X):
        """Initialize the search index."""
        lens = (X ** 2).sum(-1)  # precompute (squared) length of each vector
        if self._metric == 'angular':
            # normalize index vectors to unit length
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!

    def query(self, v, n):
        """Find indices of `n` most similar vectors from the index to query vector `v`."""
        v = numpy.ascontiguousarray(v, dtype=self._precision)
        # use same precision for query as for index
        # HACK we ignore query length as that's a constant not affecting the
        # final ordering
        if self._metric == 'angular':
            # argmax_a cossim(a, b) = argmax_a dot(a, b) / |a||b| = argmin_a -dot(a, b)
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            # argmin_a (a - b)^2 = argmin_a a^2 - 2ab + b^2 = argmin_a a^2 - 2ab
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        # partition-sort by distance, get `n` closest
        indices = numpy.argpartition(dists, n)[:n]
        # sort `n` closest into correct order
        return sorted(indices, key=lambda index: dists[index])


class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(
            INDEX_DIR, "youtube_n2_M%d_efCon%d_n_thread%s" %
            (m, ef_construction, n_threads))
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d" % (
            m, ef_construction, n_threads, ef_search)
        self._metric = metric

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        from n2 import HnswIndex
        if self._metric == 'euclidean':
            self._n2 = HnswIndex(X.shape[1], 'L2')
        else:
            self._n2 = HnswIndex(X.shape[1])
        if os.path.exists(self._index_name):
            logging.debug("Loading index from file")
            self._n2.load(self._index_name)
        else:
            logging.debug(
                "Index file is not exist: {0}".format(
                    self._index_name))
            logging.debug("Start fitting")

            for i, x in enumerate(X):
                self._n2.add_data(x.tolist())
            self._n2.build(
                m=self._m,
                max_m0=self._m0,
                ef_construction=self._ef_construction,
                n_threads=self._n_threads)
            self._n2.save(self._index_name)

    def query(self, v, n):
        return self._n2.search_by_vector(v.tolist(), n, self._ef_search)

    def __str__(self):
        return self.name


class NmslibReuseIndex(BaseANN):
    def __init__( self, metric, method_name, index_param, save_index,query_param):
        self._nmslib_metric = {
            'angular': 'cosinesimil',
            'euclidean': 'l2'}[metric]
        self._method_name = method_name
        self._save_index = save_index
        self._index_param = index_param
        self._query_param = query_param
        self.name = 'Nmslib(method_name=%s, index_param=%s, query_param=%s)' % (
            method_name, index_param, query_param)
        self._index_name = os.path.join(
            INDEX_DIR, "youtube_nmslib_%s_%s_%s" %
            (self._method_name, metric, '_'.join(
                self._index_param)))

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        import nmslib
        self._index = nmslib.init(
            self._nmslib_metric,
            [],
            self._method_name,
            nmslib.DataType.DENSE_VECTOR,
            nmslib.DistType.FLOAT)

        for i, x in enumerate(X):
            nmslib.addDataPoint(self._index, i, x.tolist())

        if os.path.exists(self._index_name):
            logging.debug("Loading index from file")
            nmslib.loadIndex(self._index, self._index_name)
        else:
            logging.debug("Create Index")
            nmslib.createIndex(self._index, self._index_param)
            if self._save_index:
                nmslib.saveIndex(self._index, self._index_name)

        nmslib.setQueryTimeParams(self._index, self._query_param)

    def query(self, v, n):
        import nmslib
        return nmslib.knnQuery(self._index, n, v.tolist())

    def freeIndex(self):
        import nmslib
        nmslib.freeIndex(self._index)


class Annoy(BaseANN):
    def __init__(self, metric, n_trees, search_k):
        self._n_trees = n_trees
        self._search_k = search_k
        self._metric = metric
        self._index_name = os.path.join(
            INDEX_DIR, "youtube_annoy_%s_tree%d" %
            (metric, n_trees))
        self.name = 'Annoy(n_trees=%d, search_k=%d)' % (n_trees, search_k)

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        import annoy
        self._annoy = annoy.AnnoyIndex(f=X.shape[1], metric=self._metric)
        if os.path.exists(self._index_name):
            logging.debug("Loading index from file")
            self._annoy.load(self._index_name)
        else:
            logging.debug("Index file not exist start fitting!!")
            for i, x in enumerate(X):
                self._annoy.add_item(i, x.tolist())
            self._annoy.build(self._n_trees)
            self._annoy.save(self._index_name)

    def query(self, v, n):
        return self._annoy.get_nns_by_vector(v.tolist(), n, self._search_k)


def get_fn(base, args):
    fn = os.path.join(base, args.dataset)

    if args.limit != -1:
        fn += '-%d' % args.limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn


def run_algo(args, library, algo, results_fn):
    pool = multiprocessing.Pool()
    X_train, X_test = get_dataset(args.dataset, args.limit)
    pool.close()
    pool.join()

    t0 = time.time()
    algo.fit(X_train)
    build_time = time.time() - t0
    logging.debug('Built index in {0}'.format(build_time))
    best_search_time = float('inf')
    best_precision = 0.0  # should be deterministic but paranoid
    try_count = 3
    # Do multiple times to warm up page cache, use fastest
    for i in xrange(try_count):
        results = []
        search_time = 0.0
        current_query = 1
        total_queries = len(queries)
        for j in range(total_queries):
            v, correct = queries[j]
            sys.stdout.write(
                "Querying: %d / %d \r" %
                (current_query, total_queries))
            t0 = time.time()
            found = algo.query(v, GT_SIZE)
            search_time += (time.time() - t0)
            if len(found) < len(correct):
                logging.debug(
                    'found: {0}, correct: {1}'.format(
                        len(found), len(correct)))
            current_query += 1
            results.append(len(set(found).intersection(correct)))

        k = float(sum(results))
        search_time /= len(queries)
        precision = k / (len(queries) * GT_SIZE)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)
        sys.stdout.write(
            '*[%d/%d][algo: %s] search time: %s, precision: %.5f \r' %
            (i + 1, try_count, str(algo), str(search_time), precision))
        sys.stdout.write('\n')
    output = [library, algo.name, build_time, best_search_time, best_precision]
    logging.debug(str(output))
    f = open(results_fn, 'a')
    f.write('\t'.join(map(str, output)) + '\n')
    f.close()
    logging.debug('Summary: {0}'.format('\t'.join(map(str, output))))


def get_dataset(which='glove', limit=-1, random_state=3, test_size=10000):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, test_size, limit, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        logging.debug('{0} {1}'.format(X_train.shape, X_test.shape))
        return X_train, X_test
    local_fn = os.path.join('datasets', which)
    if os.path.exists(local_fn + '.gz'):
        f = gzip.open(local_fn + '.gz')
    else:
        f = open(local_fn + '.txt')

    X = []
    for i, line in enumerate(f):
        v = [float(x) for x in line.strip().split()]
        X.append(v)
        if limit != -1 and len(X) == limit:
            break

    X = numpy.vstack(X)
    import sklearn.cross_validation

    # Here Erik is most welcome to use any other random_state
    # However, it is best to use a new random seed for each major re-evaluation,
    # so that we test on a trully bind data.
    X_train, X_test = sklearn.cross_validation.train_test_split(
        X, test_size=test_size, random_state=random_state)
    X_train = X_train.astype(numpy.float)
    X_test = X_test.astype(numpy.float)
    numpy.savez(cache, train=X_train, test=X_test)
    return X_train, X_test


def get_queries(args):
    logging.debug('computing queries with correct results...')

    bf = BruteForceBLAS(args.distance)
    X_train, X_test = get_dataset(which=args.dataset, limit=args.limit)

    # Prepare queries
    bf.fit(X_train)
    queries = []
    total_queries = len(X_test)
    for x in X_test:
        correct = bf.query(x, GT_SIZE)
        queries.append((x, correct))
        sys.stdout.write(
            'computing queries %d/%d ...\r' %
            (len(queries), total_queries))

    sys.stdout.write('\n')
    return queries


def get_fn(base, args):
    fn = os.path.join(base, args.dataset)

    if args.limit != -1:
        fn += '-%d' % args.limit
    if os.path.exists(fn + '.gz'):
        fn += '.gz'
    else:
        fn += '.txt'

    d = os.path.dirname(fn)
    if not os.path.exists(d):
        os.makedirs(d)

    return fn


if __name__ == '__main__':
    global GT_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', help='Algorithm', type=str)
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--n_threads', help='Number of threads', type=int,default=20)
    args = parser.parse_args()
    args.dataset = 'youtube'
    args.distance = 'angular'
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    GT_SIZE = 100

    if not os.path.exists(YOUTUBE_DIR):
        raise IOError(
            'Please follow the instructions in the guide to download the YouTube dataset.')

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)
    logging.debug(
        'storing queries in {0} and results in {1}.'.format(
            queries_fn, results_fn))

    if not os.path.exists(queries_fn):
        queries = get_queries(args)
        with open(queries_fn, 'wb') as f:
            pickle.dump(queries, f)
    else:
        queries = pickle.load(open(queries_fn, 'rb'))

    logging.debug('got {0} queries'.format(len(queries)))

    algos = {
        'annoy': [ Annoy('angular', n_trees, search_k)
                  for n_trees in [10]
                  for search_k in [ 7, 3000, 50000, 200000, 500000]
               ], 
        'n2': [ N2(M, ef_con, args.n_threads, ef_search, 'angular') 
               for M, ef_con in [ (12, 100)] 
               for ef_search in [10, 100, 1000, 10000, 100000]
            ], 
        'nmslib': []}

    MsPostsEfs = [
        ({'M': 12,
          'post': 0,
          'indexThreadQty': args.n_threads,
          'delaunay_type': 2,
          'efConstruction': 100,
          },
         [10, 100, 1000, 10000, 100000],
         ),
    ]

    for oneCase in MsPostsEfs:
        for ef in oneCase[1]:
            params = ['%s=%s' % (k, str(v)) for k, v in oneCase[0].items()]
            algos['nmslib'].append(
                NmslibReuseIndex( 'angular', 'hnsw', params, True, ['ef=%d' % ef]))

    algos_flat = []

    if args.algo:
        print('running only: %s' % str(args.algo))
        algos = {args.algo: algos[args.algo]}

    for library in algos.keys():
        for algo in algos[library]:
            algos_flat.append((library, algo))

    random.shuffle(algos_flat)
    logging.debug('order: %s' % str([a.name for l, a in algos_flat]))

    for library, algo in algos_flat:
        logging.debug(algo.name)
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(
            target=run_algo, args=(
                args, library, algo, results_fn))
        p.start()
        p.join()
