import pathlib
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


def layer_similarity(folder, lang_pairs):
    gradient_l1 = {}
    gradient_l2 = {}
    for file in sorted(folder.glob('[0-9]*.pt'))[:40]:
        gradient = torch.load(file)
        for key in gradient[lang_pairs[0]].keys():
            gradient_l1[key] = gradient[lang_pairs[0]][key].float() + gradient_l1.get(key, 0.)
            gradient_l2[key] = gradient[lang_pairs[1]][key].float() + gradient_l2.get(key, 0.)
    for key in gradient_l1.keys():
        cos_similarity = cosine_similarity([gradient_l1[key].numpy(), gradient_l2[key].numpy()])[0, 1]
        print(cos_similarity)


def encoder_similarity(gradients, lang_pairs):
    gradient_l1 = gradients[lang_pairs[0]]
    gradient_l2 = gradients[lang_pairs[1]]
    gradient_l1 = torch.cat([v for k, v in sorted(gradient_l1.items(), key=lambda p: p[0]) if 'encoder.layers.1' in k])
    gradient_l2 = torch.cat([v for k, v in sorted(gradient_l2.items(), key=lambda p: p[0]) if 'encoder.layers.1' in k])
    cos_similarity = cosine_similarity([gradient_l1.numpy(), gradient_l2.numpy()])[0, 1]
    print(cos_similarity)


def rev_lang(lang_pairs):
    return_value = []
    for lang_pair in lang_pairs:
        tgt, src = lang_pair.split('-')
        return_value.append("{}-{}".format(src, tgt))
    return return_value


def pari_wise_sim():
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/es.fr/multi/2to1')
    lang_pairs = ['en-es', 'en-fr']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/es.zh/multi/2to1')
    lang_pairs = ['en-es', 'en-zh']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/fr.zh/multi/2to1')
    lang_pairs = ['en-zh', 'en-fr']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)



def multi_lang_sim():
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/13/multi/13to1')
    lang_pairs = ['fr-en', 'de-en', 'lv-en', 'lt-en', 'et-en', 'fi-en', 'id-en', 'ms-en', 'pl-en', 'cs-en', 'uk-en', 'ru-en']
    lang_pairs = lang_pairs if folder.name == '13to1' else rev_lang(lang_pairs)
    gradients = torch.load(folder / 'average_grad.pt')

    cluster = AgglomerativeClustering(
        linkage='average', affinity='cosine', n_clusters=2, compute_distances=True)
    clustering_input = []
    lang_pairs = ['fr-en', 'de-en', 'lv-en', 'lt-en', 'et-en', 'fi-en', 'id-en', 'ms-en', 'pl-en', 'cs-en', 'uk-en', 'ru-en']
    lang_pairs = ['fr-en', 'de-en', 'id-en', 'ms-en', 'uk-en', 'ru-en']
    lang_pairs = ['fr-en', 'de-en', 'lv-en', 'lt-en', 'et-en', 'fi-en']
    keys, values = zip(*gradients.items())
    print(keys, values)
    exit()
    for lang_pair, value in gradients.items():
        if lang_pair not in lang_pairs:
            continue
        for key, grad in value.items():
            if key == 'encoder.layers.1':
                clustering_input.append(grad.numpy())
    clustering_input = np.array(clustering_input)
    labels = cluster.fit_predict(clustering_input)
    print(cluster.distances_)
    print(labels)
    plot_dendrogram(cluster)


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    print(linkage_matrix)


def main():
    # pari_wise_sim()
    multi_lang_sim()


if __name__ == '__main__':
    main()
