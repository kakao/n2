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


def visualize_all(args, data, fname):
    # fig = plt.figure(figsize=(8, 12))
    fig = plt.figure(figsize=(12, 24))
    if args.title is not None:
        fig.suptitle(args.title, fontsize=20)

    ax1 = fig.add_subplot(4, 1, 1)
    draw_build_time(ax1, data)

    ax2 = fig.add_subplot(4, 1, 2)
    draw_index_size(ax2, data)

    ax3 = fig.add_subplot(4, 1, 3)
    draw_accuracy_elapsed(ax3, data)

    ax4 = fig.add_subplot(4, 1, 4)
    draw_accuracy_elapsed(ax4, data, limit=0.8)

    fig.tight_layout()
    if args.title is not None:
        fig.subplots_adjust(top=0.95)
    fig.savefig(fname)


def visualize_auccracy_only(args, data, fname):
    fig = plt.figure(figsize=(12, 6))
    if args.title is not None:
        fig.suptitle(args.title, fontsize=20)

    ax = fig.add_subplot(1, 1, 1)
    draw_accuracy_elapsed(ax, data)

    fig.tight_layout()
    if args.title is not None:
        fig.subplots_adjust(top=0.89)
    fig.savefig(fname)


def visualize(args, data, fname):
    if args.accuracy_only:
        visualize_auccracy_only(args, data, fname)
    else:
        visualize_all(args, data, fname)


def draw_build_time(ax, data):
    ax.set_title('build time', fontsize=15)
    keys = data.keys()
    y = [max([t for _, _, t, _ in data[k]]) for k in keys]
    x = list(range(len(y)))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=12)
    ax.set_ylabel('elapsed(sec)', fontsize=12)


def draw_accuracy_elapsed(ax, data, limit=0.5):
    ax.set_title('Recall-elasped sec tradeoff', fontsize=15)
    for library, values in data.items():
        values = sorted(filter(lambda x: x[0] > limit, values))
        ax.plot([acc for acc, _, _, _ in values], [elapsed for _, elapsed, _, _ in values],
                label=library, marker='o', markersize=3)
    ax.set_xlabel('accuracy', fontsize=12)
    ax.set_ylabel('elapsed(log scale)', fontsize=12)
    ax.legend(loc='best')


def draw_index_size(ax, data):
    ax.set_title('index size(kb)', fontsize=15)
    keys = data.keys()
    y = [min([kb for _, _, _, kb in data[k]]) for k in keys]
    x = list(range(len(y)))
    ax.bar(x, y)
    ax.set_xticks(x)
    ax.set_xticklabels(keys, fontsize=12)
    ax.set_ylabel('kb', fontsize=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', help='title of figure')
    parser.add_argument('--accuracy_only', help='draw accuracy only', action='store_true')
    parser.add_argument('result', help='result file path')
    args = parser.parse_args()

    if not os.path.exists(args.result):
        raise ValueError("Wrong result file path")

    visualize(args, parse(args.result), os.path.splitext(args.result)[0] + '.png')
