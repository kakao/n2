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

try:
    xrange
except NameError:
    xrange = range

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

from n2 import HnswIndex

n2_logger = logging.getLogger("n2_benchmark")
n2_logger.setLevel(logging.INFO)

# Set resource limits to prevent memory bombs
memory_limit = 12 * 2**30
soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
if soft == resource.RLIM_INFINITY or soft >= memory_limit:
    n2_logger.info('resetting memory limit from {0} to {1}. '.format(soft, memory_limit))
    resource.setrlimit(resource.RLIMIT_DATA, (memory_limit, hard))

INDEX_DIR='indices'
DATA_DIR = './datasets/'
GLOVE_DIR = DATA_DIR + 'glove.txt'
SIFT_DIR = DATA_DIR + 'sift.txt'
YOUTUBE_DIR = DATA_DIR + 'youtube.txt'

class BaseANN(object):
    def use_threads(self):
        return True

class BruteForceBLAS(BaseANN):
    """kNN search that uses a linear scan = brute force."""
    def __init__(self, metric, precision=numpy.float32):
        if metric not in ('angular', 'euclidean'):
            raise NotImplementedError("BruteForceBLAS doesn't support metric %s" % metric)
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
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!

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
        else:
            assert False, "invalid metric"  # shouldn't get past the constructor!
        indices = numpy.argpartition(dists, n)[:n]  # partition-sort by distance, get `n` closest
        return sorted(indices, key=lambda index: dists[index])  # sort `n` closest into correct order

class N2(BaseANN):
    def __init__(self, m, ef_construction, n_threads, ef_search, metric):
        self._m = m
        self._m0 = m * 2
        self._ef_construction = ef_construction
        self._n_threads = n_threads
        self._ef_search = ef_search
        self._index_name = os.path.join(INDEX_DIR, "n2_%s_M%d_efCon%d_n_thread%s" % (args.dataset, m, ef_construction, n_threads))
        self.name = "N2_M%d_efCon%d_n_thread%s_efSearch%d" % (m, ef_construction, n_threads, ef_search)
        self._metric = metric

        d = os.path.dirname(self._index_name)
        if not os.path.exists(d):
            os.makedirs(d)

    def fit(self, X):
        if self._metric == 'euclidean':
            self._n2 = HnswIndex(X.shape[1], 'L2')
        else:
            self._n2 = HnswIndex(X.shape[1])
        if os.path.exists(self._index_name):
            n2_logger.info("Loading index from file")
            self._n2.load(self._index_name)
        else:
            n2_logger.info("Index file is not exist: {0}".format(self._index_name))
            n2_logger.info("Start fitting")

            for i, x in enumerate(X):
                self._n2.add_data(x.tolist())
            self._n2.build(m=self._m, max_m0=self._m0, ef_construction=self._ef_construction, n_threads=self._n_threads)
            self._n2.save(self._index_name)

    def query(self, v, n):
         return self._n2.search_by_vector(v.tolist(), n, self._ef_search)

    def __str__(self):
         return self.name

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
    n2_logger.info('Built index in {0}'.format(build_time))
    best_search_time = float('inf')
    best_precision = 0.0 # should be deterministic but paranoid
    try_count = 3
    for i in xrange(try_count): # Do multiple times to warm up page cache, use fastest
        results = []
        search_time = 0.0
        current_query = 1
        total_queries = len(queries)
        for j in range(total_queries):
            v, correct = queries[j]
            sys.stdout.write("Querying: %d / %d \r" % (current_query, total_queries))
            t0 = time.time()
            found = algo.query(v, GT_SIZE)
            search_time += (time.time() - t0)
            if len(found) < len(correct):
                n2_logger.info('found: {0}, correct: {1}'.format(len(found), len(correct)))
            current_query += 1
            results.append(len(set(found).intersection(correct)))

        k = float(sum(results))
        search_time /= len(queries)
        precision = k / (len(queries) * GT_SIZE)
        best_search_time = min(best_search_time, search_time)
        best_precision = max(best_precision, precision)
        sys.stdout.write('*[%d/%d][algo: %s] search time: %s, precision: %.5f \r' % (i+1, try_count, str(algo), str(search_time), precision))
        sys.stdout.write('\n')
    output = [library, algo.name, build_time, best_search_time, best_precision]
    n2_logger.info(str(output))
    f = open(results_fn, 'a')
    f.write('\t'.join(map(str, output)) + '\n')
    f.close()
    n2_logger.info('Summary: {0}'.format('\t'.join(map(str, output))))

def get_dataset(which='glove', limit=-1, random_state = 3, test_size = 10000):
    cache = 'queries/%s-%d-%d-%d.npz' % (which, test_size, limit, random_state)
    if os.path.exists(cache):
        v = numpy.load(cache)
        X_train = v['train']
        X_test = v['test']
        n2_logger.info('{0} {1}'.format(X_train.shape, X_test.shape))
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
    X_train, X_test = sklearn.cross_validation.train_test_split(X, test_size=test_size, random_state=random_state)
    X_train = X_train.astype(numpy.float)
    X_test = X_test.astype(numpy.float)
    numpy.savez(cache, train=X_train, test=X_test)
    return X_train, X_test

def get_queries(args):
    n2_logger.info('computing queries with correct results...')

    bf = BruteForceBLAS(args.distance)
    X_train, X_test = get_dataset(which=args.dataset, limit=args.limit)

    # Prepare queries
    bf.fit(X_train)
    queries = []
    total_queries = len(X_test)
    for x in X_test:
        correct = bf.query(x, GT_SIZE)
        queries.append((x, correct))
        sys.stdout.write('computing queries %d/%d ...\r' % (len(queries), total_queries))

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

def download_file(url, dst):
    file_name = url.split('/')[-1]
    with closing(urlopen(url)) as res:
        with open(dst+"/"+file_name, 'wb') as f:
            file_size = int(res.headers["Content-Length"])
            sys.stdout.write("Downloading datasets %s\r" % (file_name))

            file_size_dl = 0
            block_sz = 10240
            while True:
                buffer = res.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                sys.stdout.write("Downloading datasets %s: %d / %d bytes\r" % (file_name, file_size_dl, file_size))

        sys.stdout.write('\n')

if __name__ == '__main__':
    global GT_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument('--distance', help='Distance metric', default='angular')
    parser.add_argument('--dataset', help='Which dataset',  default='glove')
    parser.add_argument('--limit', help='Limit', type=int, default=-1)
    parser.add_argument('--ef_con', help='ef_con', type=int, default=100)
    parser.add_argument('--ef_search', help='ef_search', type=int, default=800)
    parser.add_argument('--M', help='M', type=int, default=10)
    parser.add_argument('--n_threads', help='Number of threads', type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if args.dataset == 'glove':
        GT_SIZE = 10
    elif args.dataset == 'sift':
        GT_SIZE = 10
    elif args.dataset == 'youtube':
        GT_SIZE = 100
    else:
        print('Invalid dataset: {}'.format(args.dataset))
        exit(0)
    print('* GT size: {}'.format(GT_SIZE))

    if args.dataset == 'glove' and not os.path.exists(GLOVE_DIR):
        download_file("https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz", "datasets")
        with gzip.open('datasets/glove.twitter.27B.100d.txt.gz', 'rb') as f_in, open('datasets/glove.twitter.27B.100d.txt', 'w') as f_out:
            shutil.copyfileobj(f_in, f_out)
        subprocess.call("cut -d \" \" -f 2- datasets/glove.twitter.27B.100d.txt > datasets/glove.txt", shell=True)
 
    if args.dataset == 'sift' and not os.path.exists(SIFT_DIR):
        download_file("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", "datasets")
        with tarfile.open("datasets/sift.tar.gz") as t:
            t.extractall(path="datasets")
        subprocess.call("python datasets/convert_texmex_fvec.py datasets/sift/sift_base.fvecs >> datasets/sift.txt", shell=True)

    if args.dataset == 'youtube' and not os.path.exists(YOUTUBE_DIR):
        raise IOError('Please follow the instructions in the guide to download the YouTube dataset.')

    results_fn = get_fn('results', args)
    queries_fn = get_fn('queries', args)
    logging.info('storing queries in {0} and results in {1}.'.format(queries_fn, results_fn))

    if not os.path.exists(queries_fn):
        queries = get_queries(args)
        with open(queries_fn, 'wb') as f:
            pickle.dump(queries, f)
    else:
        queries = pickle.load(open(queries_fn, 'rb'))

    logging.info('got {0} queries'.format(len(queries)))

    configs = [N2(args.M, args.ef_con, args.n_threads, args.ef_search, args.distance)]

    algos_flat = []
    for c in configs:
        algos_flat.append(('n2', c))
    for library, algo in algos_flat:
        logging.info(algo.name)
        # Spawn a subprocess to force the memory to be reclaimed at the end
        p = multiprocessing.Process(target=run_algo, args=(args, library, algo, results_fn))
        p.start()
        p.join()
   
