import os
import sys
import shutil
import tarfile
import argparse
import subprocess
import gzip
from contextlib import closing

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen


DATA_DIR = './datasets/'
GLOVE_FILE = DATA_DIR + 'glove.txt'
SIFT_FILE = DATA_DIR + 'sift.txt'
YOUTUBE_FILE = DATA_DIR + 'youtube.txt'


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Which dataset',  default='glove')
    args = parser.parse_args()

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if args.dataset == 'glove' and not os.path.exists(GLOVE_FILE):
        download_file("https://s3-us-west-1.amazonaws.com/annoy-vectors/glove.twitter.27B.100d.txt.gz", "datasets")
        with gzip.open('datasets/glove.twitter.27B.100d.txt.gz', 'rb') as f_in, open('datasets/glove.twitter.27B.100d.txt', 'w') as f_out:
            shutil.copyfileobj(f_in, f_out)
        subprocess.call("cut -d \" \" -f 2- datasets/glove.twitter.27B.100d.txt > datasets/glove.txt", shell=True)

    if args.dataset == 'sift' and not os.path.exists(SIFT_FILE):
        download_file("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz", "datasets")
        with tarfile.open("datasets/sift.tar.gz") as t:
            t.extractall(path="datasets")
        subprocess.call("python datasets/convert_texmex_fvec.py datasets/sift/sift_base.fvecs >> datasets/sift.txt", shell=True)

    if args.dataset == 'youtube' and not os.path.exists(YOUTUBE_FILE):
        raise IOError('Please follow the instructions in the guide to download the YouTube dataset.')
