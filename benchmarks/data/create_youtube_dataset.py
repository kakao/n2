import sys
sys.path.append('..')

import h5py
import numpy as np

from sklearn.model_selection import train_test_split

from metrics import metrics as pd


class BruteForceBLAS():
    def __init__(self, metric, precision=np.float32):
        self._metric = metric
        self._precision = precision

    def fit(self, X):
        lens = (X ** 2).sum(-1)
        if self._metric == 'angular':
            X /= np.sqrt(lens)[..., np.newaxis]
            self.index = np.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = np.ascontiguousarray(X, dtype=self._precision)
            self.lengths = np.ascontiguousarray(lens, dtype=self._precision)
        elif self._metric == 'dot':
            self.index = np.ascontiguousarray(X, dtype=self._precision)

    def query_with_distances(self, v, n):
        v = np.ascontiguousarray(v, dtype=self.index.dtype)
        if self._metric == 'angular' or self._metric == 'dot':
            dists = -np.dot(self.index, v)
        elif self._metric == 'euclidean':
            dists = self.lengths - 2 * np.dot(self.index, v)
        else:
            assert False, "invalid metric"
        indices = np.argpartition(dists, n)[:n]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric]['distance'](ep, ev))
        return map(fix, indices)


def write_output(train, test, fn, metric, point_type='float', count=100):
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = metric
    f.attrs['point_type'] = point_type
    f.create_dataset('train', (len(train), len(train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    bf = BruteForceBLAS(metric, precision=train.dtype)
    bf.fit(train)
    for i, x in enumerate(test):
        sys.stderr.write('computing queries %d/%d ...\r' % (i+1, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    sys.stderr.write('\n')
    f.close()


def make_dataset(dataset, metrics):
    X = []
    for line in open(f'{dataset}.txt'):
        v = [float(x) for x in line.strip().split()]
        X.append(np.array(v))
    dim = len(X[0])
    X_train, X_test = train_test_split(X, test_size=10000, random_state=1)

    for metric in metrics:
        write_output(np.array(X_train), np.array(X_test), f'{dataset}-{dim}-{metric}.hdf5', metric)

if __name__ == '__main__':
    for dataset in ['youtube1m', 'youtube']:
        make_dataset(dataset, ['angular', 'dot'])
