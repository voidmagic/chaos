import pathlib
import torch
from sklearn.metrics.pairwise import cosine_similarity


def layer_similarity(folder, lang_pairs):
    gradient_l1 = {}
    gradient_l2 = {}
    for file in sorted(folder.glob('[0-9]*.pt'))[-1:]:
        gradient = torch.load(file)
        for key in gradient[lang_pairs[0]].keys():
            gradient_l1[key] = gradient[lang_pairs[0]][key].float() + gradient_l1.get(key, 0.)
            gradient_l2[key] = gradient[lang_pairs[1]][key].float() + gradient_l2.get(key, 0.)
    for key in gradient_l1.keys():
        cos_similarity = cosine_similarity([gradient_l1[key].numpy(), gradient_l2[key].numpy()])[0, 1]
        print(cos_similarity)


def encoder_similarity(folder, lang_pairs):
    gradient_l1 = {}
    gradient_l2 = {}
    for file in sorted(folder.glob('[0-9]*.pt'))[:]:
        gradient = torch.load(file)
        for key in gradient[lang_pairs[0]].keys():
            if 'encoder' in key:
                gradient_l1[key] = gradient[lang_pairs[0]][key].float() + gradient_l1.get(key, 0.)
                gradient_l2[key] = gradient[lang_pairs[1]][key].float() + gradient_l2.get(key, 0.)
    gradient_l1 = torch.cat([v for _, v in sorted(gradient_l1.items(), key=lambda p: p[0])])
    gradient_l2 = torch.cat([v for _, v in sorted(gradient_l2.items(), key=lambda p: p[0])])
    cos_similarity = cosine_similarity([gradient_l1.numpy(), gradient_l2.numpy()])[0, 1]
    print(cos_similarity)


def rev_lang(lang_pairs):
    return_value = []
    for lang_pair in lang_pairs:
        tgt, src = lang_pair.split('-')
        return_value.append("{}-{}".format(src, tgt))
    return return_value


def pari_wise_sim():
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/es.fr/multi/1to2')
    lang_pairs = ['en-es', 'en-fr']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/es.zh/multi/1to2')
    lang_pairs = ['en-es', 'en-zh']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/fr.zh/multi/1to2')
    lang_pairs = ['en-zh', 'en-fr']
    lang_pairs = lang_pairs if folder.name == '1to2' else rev_lang(lang_pairs)
    print(lang_pairs)
    layer_similarity(folder, lang_pairs)


def main():
    folder = pathlib.Path('/data/qwang/018-auto-share/003-models/403-opus-analyze/7/multi/7to1')
    lang_pairs = ['fr-en', 'es-en', 'de-en', 'lv-en', 'it-en', 'et-en', 'fi-en']
    lang_pairs = lang_pairs if folder.name == '7to1' else rev_lang(lang_pairs)
    for i in range(len(lang_pairs)):
        for j in range(i+1, len(lang_pairs)):
            print([lang_pairs[i], lang_pairs[j]])
            encoder_similarity(folder, [lang_pairs[i], lang_pairs[j]])


if __name__ == '__main__':
    main()
