import os
import math
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def parse(fname):
    data = defaultdict(list)
    for line in open(fname):
        library, _, build_time, search_elapsed, accuracy = line.strip().split('\t')
        data[library].append((float(accuracy), math.log(float(search_elapsed)), float(build_time)))
    return data


def visualize(data):
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 1, 1)
    draw_build_time(ax1, data)

    ax2 = fig.add_subplot(2, 1, 2)
    draw_accuracy_elapsed(ax2, data)

    plt.show()


def draw_build_time(ax, data):
    keys = data.keys()
    y = [max([t for _, _, t in data[k]]) for k in keys]
    x = list(range(len(y)))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel('elapsed(sec)')


def draw_accuracy_elapsed(ax, data):
    for library, values in data.items():
        values = filter(lambda x: x[0] > 0.5, values)
        values.sort()

        ax.plot([acc for acc, _, _ in values], [elapsed for _, elapsed, _ in values], label=library)

    ax.set_xlabel('accuracy')
    ax.set_ylabel('elapsed(log scale)')
    ax.legend(loc='best')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result file path')
    args = parser.parse_args()

    if not os.path.exists(args.result):
        raise ValueError("Wrong result file path")

    visualize(parse(args.result))
