import os
import math
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict


def parse(fname):
    data = defaultdict(list)
    for line in open(fname):
        library, _, build_time, search_elapsed, accuracy, index_size_kb = line.strip().split('\t')
        data[library].append((float(accuracy), math.log(float(search_elapsed)), float(build_time), float(index_size_kb)))
    return data


def visualize(data, fname):
    fig = plt.figure(figsize=(8, 12))

    ax1 = fig.add_subplot(4, 1, 1)
    draw_build_time(ax1, data)

    ax2 = fig.add_subplot(4, 1, 2)
    draw_index_size(ax2, data)

    ax3 = fig.add_subplot(4, 1, 3)
    draw_accuracy_elapsed(ax3, data)

    ax4 = fig.add_subplot(4, 1, 4)
    draw_accuracy_elapsed(ax4, data, limit=0.8)

    fig.tight_layout()
    fig.savefig(fname)


def draw_build_time(ax, data):
    ax.set_title('build time')
    keys = data.keys()
    y = [max([t for _, _, t, _ in data[k]]) for k in keys]
    x = list(range(len(y)))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel('elapsed(sec)')


def draw_accuracy_elapsed(ax, data, limit=0.5):
    ax.set_title('Recall-elasped sec tradeoff')
    for library, values in data.items():
        values = sorted(filter(lambda x: x[0] > limit, values))
        ax.plot([acc for acc, _, _, _ in values], [elapsed for _, elapsed, _, _ in values], label=library)
    ax.set_xlabel('accuracy')
    ax.set_ylabel('elapsed(log scale)')
    ax.legend(loc='best')


def draw_index_size(ax, data):
    ax.set_title('index size(kb)')
    keys = data.keys()
    y = [min([kb for _, _, _, kb in data[k]]) for k in keys]
    x = list(range(len(y)))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(keys)
    ax.set_ylabel('kb')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='result file path')
    args = parser.parse_args()

    if not os.path.exists(args.result):
        raise ValueError("Wrong result file path")

    visualize(parse(args.result), os.path.splitext(args.result)[0] + '.png')
