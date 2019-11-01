import os
import sys
import gzip
import shutil
import tarfile
import argparse
import subprocess
from contextlib import closing

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve  # Python 3


DATASETS = [
    'fashion-mnist-784-euclidean',
    'gist-960-euclidean',
    'glove-25-angular',
    'glove-50-angular',
    'glove-100-angular',
    'glove-200-angular',
    'mnist-784-euclidean',
    'random-xs-20-euclidean',
    'random-s-100-euclidean',
    'random-xs-20-angular',
    'random-s-100-angular',
    'sift-128-euclidean',
    'nytimes-256-angular',
    'youtube-40-angular',
    'youtube1m-40-angular',
]


def download(src, dst):
    if not os.path.exists(dst):
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s.hdf5' % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        raise IOError("Cannot download %s" % url)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset',  choices=DATASETS)
    args = parser.parse_args()

    get_dataset(args.dataset)
