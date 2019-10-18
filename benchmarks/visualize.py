import os
import math
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def parse(fname):
    data = defaultdict(list)
    for line in open(fname):
        library, _, _, search_elapsed, accuracy = line.strip().split('\t')
        data[library].append((float(accuracy), math.log(float(search_elapsed))))
    return data


def visualize(data):
    fig, ax = plt.subplots()

    for library, values in data.items():
        values.sort()
        ax.plot([acc for acc, _ in values], [elapsed for _, elapsed in values], label=library)

    ax.set_xlabel('accuracy')
    ax.set_ylabel('elapsed(log scale)')
    ax.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result file path')
    args = parser.parse_args()

    if not os.path.exists(args.result):
        raise ValueError("Wrong result file path")

    visualize(parse(args.result))
