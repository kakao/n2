import os
import re
import argparse
from collections import defaultdict


def parse(args):
    data = defaultdict(dict)
    for line in open(args.fname):
        library, algo, _, search_elapsed, accuracy, _ = line.strip().split('\t')
        data[library.split(' ')[0]][algo] = float(search_elapsed), float(accuracy)
    return data[args.base_lib], data[args.target_lib]


def compare(base, target):
    def natural_sort(l):
        def alphanum_key(key):
            def convert(text):
                return int(text) if text.isdigit() else text.lower()
            return [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    print('algo\tsearch_elapsed_seconds(negative values are better)\taccuracy(positive values are better)')
    for key in natural_sort(target.keys()):
        if key in base:
            print('%s\t%s' % (key, '\t'.join(str(round((z[0] - z[1]) * 100 / z[1], 2)) + ' %'
                                             for z in zip(target[key], base[key]))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_lib', help='base library to compare')
    parser.add_argument('--target_lib', help='target library to compare')
    parser.add_argument('fname', help='result file path')
    args = parser.parse_args()

    if not os.path.exists(args.fname):
        raise ValueError("Wrong result file path")

    compare(*parse(args))
