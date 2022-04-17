import argparse
import collections
import pathlib
import torch

import numpy as np
np.set_printoptions(precision=5, suppress=True)

parser = argparse.ArgumentParser()
parser.add_argument("--weight-path", required=True, type=str)
parser.add_argument("--tokens", default="all", type=str, help="all, first, last, mid")
args = parser.parse_args()


def overall(weights):
    overall_weight = 0
    for i in range(len(weights)):
        weight = weights[i][1]
        overall_weight += weight.mean(axis=0)
    value = overall_weight / len(weights)
    for i, value_ in enumerate(value):
        for j, _ in enumerate(value_):
            print(value[i][j], end='\t')
        print()


def per_layer(weights):
    per_layer_weight = collections.defaultdict(int)
    for i in range(len(weights)):
        weight = weights[i][1]
        per_layer_weight[weights[i][0]] += weight.mean(axis=0)

    for key, value in sorted(per_layer_weight.items()):
        print(key)
        value = value / len(weights) * len(per_layer_weight.keys())
        for i, value_ in enumerate(value):
            for j, _ in enumerate(value_):
                print(value[i][j], end='\t')
            print()


def main():
    all_weights = []
    for path in pathlib.Path(args.weight_path).glob('*'):
        layer_idx = path.name.split('.')[0]
        weight = torch.load(path).numpy()
        length = weight.shape[0]
        if args.tokens == "all" or length < 5:
            pass
        elif args.tokens == "first":
            weight = weight[:int(0.25 * length)]
        elif args.tokens == "last":
            weight = weight[int(0.425 * length):int(0.725 * length)]
        elif args.tokens == "mid":
            weight = weight[int(0.25 * length):]
        all_weights.append((layer_idx, weight))

    per_layer(all_weights)


if __name__ == '__main__':
    main()
