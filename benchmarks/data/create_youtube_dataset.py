import sys

import h5py
import numpy
from sklearn.model_selection import train_test_split


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

    def query(self, v, n):
        v = numpy.ascontiguousarray(v, dtype=self._precision)
        if self._metric == 'angular':
            dists = -numpy.dot(self.index, v)
        elif self._metric == 'euclidean':
            dists = self.lengths - 2 * numpy.dot(self.index, v)
        indices = numpy.argpartition(dists, n)[:n]
        return sorted(indices, key=lambda index: dists[index])


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


def youtube(out_fn):
    fn = './data/youtube.txt'
    X = []
    for line in open(fn):
        v = [float(x) for x in line.strip().split()]
        X.append(numpy.array(v))

    X_train, X_test = train_test_split(X, test_size=10000, random_state=1)
    write_output(numpy.array(X_train), numpy.array(X_test), out_fn, 'angular')



if __name__ == '__main__':
    out_fn = os.path.join('data', 'youtube-40-angular.hdf5')
    youtube(out_fn)
