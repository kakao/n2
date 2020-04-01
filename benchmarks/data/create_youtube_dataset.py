import sys
sys.path.append('..')

import h5py
import numpy

from sklearn.model_selection import train_test_split

from distance import metrics as pd


class BruteForceBLAS():
    def __init__(self, metric, precision=numpy.float32):
        self._metric = metric
        self._precision = precision

    def fit(self, X):
        lens = (X ** 2).sum(-1)
        if self._metric == 'angular':
            X /= numpy.sqrt(lens)[..., numpy.newaxis]
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
        elif self._metric == 'euclidean':
            self.index = numpy.ascontiguousarray(X, dtype=self._precision)
            self.lengths = numpy.ascontiguousarray(lens, dtype=self._precision)

    def query_with_distances(self, v, n):
        v = numpy.ascontiguousarray(v, dtype=self.index.dtype)
        if self._metric == 'angular':
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        else:
            assert False, "invalid metric"
        indices = numpy.argpartition(dists, n)[:n]

        def fix(index):
            ep = self.index[index]
            ev = v
            return (index, pd[self._metric]['distance'](ep, ev))
        return map(fix, indices)


def write_output(train, test, fn, distance, point_type='float', count=100):
    f = h5py.File(fn, 'w')
    f.attrs['distance'] = distance
    f.attrs['point_type'] = point_type
    f.create_dataset('train', (len(train), len(train[0])), dtype=train.dtype)[:] = train
    f.create_dataset('test', (len(test), len(test[0])), dtype=test.dtype)[:] = test
    neighbors = f.create_dataset('neighbors', (len(test), count), dtype='i')
    distances = f.create_dataset('distances', (len(test), count), dtype='f')
    bf = BruteForceBLAS(distance, precision=train.dtype)
    bf.fit(train)
    for i, x in enumerate(test):
        sys.stderr.write('computing queries %d/%d ...\r' % (i+1, len(test)))
        res = list(bf.query_with_distances(x, count))
        res.sort(key=lambda t: t[-1])
        neighbors[i] = [j for j, _ in res]
        distances[i] = [d for _, d in res]
    sys.stderr.write('\n')
    f.close()


def youtube(in_fn, out_fn):
    X = []
    for line in open(in_fn):
        v = [float(x) for x in line.strip().split()]
        X.append(numpy.array(v))

    X_train, X_test = train_test_split(X, test_size=10000, random_state=1)
    write_output(numpy.array(X_train), numpy.array(X_test), out_fn, 'angular')


if __name__ == '__main__':
    youtube('youtube1m.txt', 'youtube1m-40-angular.hdf5')
    youtube('youtube.txt', 'youtube-40-angular.hdf5')
