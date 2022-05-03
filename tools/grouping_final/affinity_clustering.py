import copy
import math

import tqdm
import numpy as np

root_path = "/home/qwang/027-optim/003-models/702-multilingual-tedm-1"
log_path = f"{root_path}/log.txt"


all_languages = set()

with open(log_path) as f:
    for line in f:
        if "fairseq.data.multilingual.multilingual_data_manager" not in line or "src_langtok" not in line:
            continue
        if "| train_inner |" in line:
            break
        lang_key = line.split()[7]
        all_languages.add(lang_key)


def remove_common(a_list):
    i, j = 0, len(a_list[0])
    while i < j:
        if len(set([s[i] for s in a_list])) > 1:
            break
        i += 1
    while i < j:
        if len(set([s[j-1] for s in a_list])) > 1:
            break
        j -= 1

    a_list = [s[i:j] for s in a_list]
    return a_list, i, j


all_languages, start, end = remove_common(list(all_languages))

affinity_matrix = {lang1: {lang2: [] for lang2 in all_languages} for lang1 in all_languages}


diff_sum = 0
valid_lines = []
with open(log_path) as f:
    for line in tqdm.tqdm(f.readlines()):
        if "Affinity" not in line or "modules.preprocessing.affinity.affinity_task_3" not in line:
            continue
        valid_lines.append(line)

num_steps, step_start, step_end = 0, int(0 * len(valid_lines)), int(1 * len(valid_lines))
for line in tqdm.tqdm(valid_lines[step_start:step_end]):
    language = line.strip().split(" | ")[4][start:end]
    target = line.strip().split(" | ")[6][start:end]
    loss_before = float(line.strip().split(" | ")[7])
    loss_after = float(line.strip().split(" | ")[8])
    loss_differ = float(line.strip().split(" | ")[9])
    diff_sum += (loss_before - loss_after)
    affinity_matrix[target][language].append(loss_before - loss_after)

print("\t" + "\t".join(sorted(affinity_matrix.keys())))
for key1 in sorted(affinity_matrix.keys()):
    print(key1, end="\t")
    for key2 in sorted(affinity_matrix[key1].keys()):
        scores = affinity_matrix[key1][key2]
        affinity_matrix[key1][key2] = np.sum(scores or [0])
        print("%.8f" % float(affinity_matrix[key1][key2]), end="\t")
    print()


affinity_dict = affinity_matrix
# clusters = {tuple([l]): [k for k in affinity_dict.keys() if affinity_dict[l][k] > 0] for l in list(affinity_dict.keys())}
# print(clusters)


def compute_score(clusters):
    over_all = 0
    for key, value in clusters.items():
        for lang in key:
            score = [affinity_dict[lang][l] for l in value if l != lang]
            score = sum(score) / len(score)
            over_all += score
    return over_all


def combine_cluster(clusters, key_1, key_2):
    new_cluster = {copy.deepcopy(k): copy.deepcopy(v) for k, v in clusters.items() if k != key_1 and k != key_2}
    new_key = key_1 + key_2
    new_value = set(new_key)
    for lang_1 in affinity_dict.keys():
        if all([affinity_dict[lang_2][lang_1] > -0.2 for lang_2 in new_key if lang_1 != lang_2]):
            new_value.add(lang_1)
    new_cluster[new_key] = list(new_value)
    return new_cluster


def top_k(key_list, value_list, k=100):

    def calculate_affinity(aux_lang):
        score = 0
        for lang in key_list:
            score += affinity_dict[lang][aux_lang]
        return score

    aux_lang_scores = {lang: calculate_affinity(lang) for lang in value_list if lang not in key_list}
    aux_lang_scores = sorted(aux_lang_scores.keys(), key=lambda key: -aux_lang_scores[key])
    return list(key_list) + aux_lang_scores[:k]


# while len(clusters) > 2:
#     result = []
#     for key_1 in clusters.keys():
#         for key_2 in clusters.keys():
#             if key_1 == key_2: continue
#             new_cluster = combine_cluster(clusters, key_1, key_2)
#             result.append((key_1, key_2, compute_score(new_cluster)))
#     result = sorted(result, key=lambda p: -p[2])
#     print(len(clusters), result[0][0], result[0][1], result[0][2])
#     clusters = combine_cluster(clusters, result[0][0], result[0][1])
#     items = list(clusters.items())
#     print("test:  ", " ".join([",".join(langs) for langs, _ in items]))
#     print("train: ", " ".join([",".join(top_k(key_langs, langs, k=100)) for key_langs, langs in items]))


pairs = "ja-ru ar-es he-tr,ja-tr pt-fr,tr-fr ru-tr,it-tr he-ja,ar-ja ko-he,tr-he ru-ar,fr-ar he-es,ja-es,it-es tr-ru,he-ru,es-ru es-it,ru-it,ja-it pt-he,ru-he,ja-he,it-he es-fr,he-fr,ru-fr ja-ar,tr-ar,ko-ar tr-pt,ja-pt,fr-pt,he-pt he-ko,ru-ko,ja-ko es-he,ar-he,fr-he pt-es,ru-es,ko-es,tr-es,fr-es es-ar,he-ar,it-ar,pt-ar it-fr,ko-fr,ar-fr,ja-fr pt-ru,ko-ru,fr-ru,it-ru,ar-ru es-tr,pt-tr,ko-tr,fr-tr,ar-tr it-ko,tr-ko,ar-ko,fr-ko,es-ko,pt-ko ru-pt,it-pt,es-pt,ar-pt,ko-pt pt-ja,es-ja,tr-ja,ru-ja,fr-ja,ko-ja,it-ja he-it,ko-it,tr-it,fr-it,ar-it,pt-it".split()
for pair in pairs:
    pair = pair.split(',')
    for task in affinity_dict:
        if all([affinity_dict[lang_2][task] > -0.02 for lang_2 in pair]) and task not in pair:
            print(task, end=",")
    print()


