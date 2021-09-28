import pathlib
from collections import defaultdict, OrderedDict

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def combine_gradients(gradients):
    result = OrderedDict()
    for lang_pair, value in gradients.items():
        current = OrderedDict()
        for key, grad in value.items():
            short_key = key[:16]
            if short_key in current:
                current[short_key] = torch.cat([current[short_key], grad])
            else:
                current[short_key] = grad
        result[lang_pair] = current
    return result


def pair_wise_sim():
    part = 'multi'
    for dire in ['1to2', '2to1']:
        for model in ['first', 'last']:
            clustering_input = [defaultdict(list), defaultdict(list), defaultdict(list)]
            folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/cs.pl/{}/{}'.format(part, dire))
            gradients = torch.load(folder / '{}_grad.pt'.format(model))
            gradients = combine_gradients(gradients)
            for lang_pair, value in gradients.items():
                for key, grad in value.items():
                    clustering_input[0][key].append(grad.numpy())

            folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/cs.lv/{}/{}'.format(part, dire))
            gradients = torch.load(folder / '{}_grad.pt'.format(model))
            gradients = combine_gradients(gradients)
            for lang_pair, value in gradients.items():
                for key, grad in value.items():
                    clustering_input[1][key].append(grad.numpy())

            folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/pl.lv/{}/{}'.format(part, dire))
            gradients = torch.load(folder / '{}_grad.pt'.format(model))
            gradients = combine_gradients(gradients)
            for lang_pair, value in gradients.items():
                for key, grad in value.items():
                    clustering_input[2][key].append(grad.numpy())

            print(part, dire, model)
            for key in clustering_input[0].keys():
                print(key, end='\t')
                for i in range(3):
                    sims = cosine_similarity(np.array(clustering_input[i][key]))
                    print(sims[0][1], end='\t')
                print()


def multi_lang_sim():
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/13/valid/13to1-parameter')
    gradients = torch.load(folder / 'average_grad.pt')

    clustering_input = []
    lang_pairs = ['fr-en', 'de-en', 'lv-en', 'lt-en', 'et-en', 'fi-en', 'id-en', 'ms-en', 'pl-en', 'cs-en', 'uk-en', 'ru-en']
    for lang_pair, value in gradients.items():
        current = []
        if lang_pair not in lang_pairs:
            continue
        for key, grad in value.items():
            if 'encoder.layers.5' in key:
                current.append(grad)
        if len(current) > 1:
            all_gradient = torch.cat(current, dim=0)
        else:
            all_gradient = current[0]
        clustering_input.append(all_gradient.numpy())

    clustering_input = np.array(clustering_input)
    cluster = AgglomerativeClustering(linkage='average', affinity='cosine', n_clusters=2, compute_distances=True)
    labels = cluster.fit_predict(clustering_input)
    # print(labels)
    # print(cluster.distances_)
    sims = cosine_similarity(clustering_input)
    for row in sims:
        for col in row:
            print(col, end='\t')
        print()


def main():
    # pair_wise_sim()
    multi_lang_sim()


if __name__ == '__main__':
    main()
