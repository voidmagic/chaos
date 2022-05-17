import collections
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

for key1 in sorted(affinity_matrix.keys()):
    for key2 in sorted(affinity_matrix[key1].keys()):
        scores = affinity_matrix[key1][key2]
        affinity_matrix[key1][key2] = np.sum(scores or [0])


res="""
1&ja-ru&it-ru,tr-ru,ja-it,it-pt,ar-ru,pt-ru,ko-ru,ja-fr,ru-fr,he-fr,fr-ru,es-ru,ar-es\\\specialrule{0em}{0.80pt}{0.80pt}
2&ar-es&es-it,ja-es,ru-es,ar-it,pt-es,tr-es,es-ar,ar-pt,ko-es,fr-pt,ar-ko,fr-es\\\specialrule{0em}{0.80pt}{0.80pt}
3&he-tr,ja-tr&fr-tr,ru-tr,es-tr,it-tr,pt-tr\\\specialrule{0em}{0.80pt}{0.80pt}
4&pt-fr,tr-fr&ar-fr,it-fr,es-fr,ko-fr\\\specialrule{0em}{0.80pt}{0.80pt}
5&ru-tr,it-tr&ar-tr,he-tr,ja-tr\\\specialrule{0em}{0.80pt}{0.80pt}
6&he-ja,ar-ja&tr-ja,ko-ja,it-ja,pt-ja\\\specialrule{0em}{0.80pt}{0.80pt}
7&ko-he,tr-he&ja-he,ru-he,es-he,ar-he,it-he\\\specialrule{0em}{0.80pt}{0.80pt}
8&ru-ar,fr-ar&tr-ar,es-ar,ko-ar,it-ar\\\specialrule{0em}{0.80pt}{0.80pt}
9&he-es,ja-es,it-es&ru-es,pt-es,ko-es,ar-es,fr-es\\\specialrule{0em}{0.80pt}{0.80pt}
10&tr-ru,he-ru,es-ru&ar-ru,pt-ru,fr-ru\\\specialrule{0em}{0.80pt}{0.80pt}
11&es-it,ru-it,ja-it&he-it,pt-it,ko-fr,fr-es,pt-es,he-it,fr-it,pt-it,tr-fr\\\specialrule{0em}{0.80pt}{0.80pt}
12&pt-he,ru-he,ja-he,it-he&ko-he,tr-he,fr-ar\\\specialrule{0em}{0.80pt}{0.80pt}
13&es-fr,he-fr,ru-fr&ar-fr,it-fr,pt-fr\\\specialrule{0em}{0.80pt}{0.80pt}
14&ja-ar,tr-ar,ko-ar&ru-ar,fr-ar,es-ar,he-ar,it-ar\\\specialrule{0em}{0.80pt}{0.80pt}
15&tr-pt,ja-pt,fr-pt,he-pt&ko-pt,it-pt,tr-it,ru-pt,es-pt\\\specialrule{0em}{0.80pt}{0.80pt}
16&he-ko,ru-ko,ja-ko&tr-ja,ja-ar,pt-ko,es-tr,ar-ko,ru-ar,tr-ko\\\specialrule{0em}{0.80pt}{0.80pt}
17&es-he,ar-he,fr-he&tr-he,it-he,es-ru,pt-he,ru-he,ko-he,ja-he\\\specialrule{0em}{0.80pt}{0.80pt}
18&pt-es,ru-es,ko-es,tr-es,fr-es&ja-es,it-es,ko-fr,ru-ar\\\specialrule{0em}{0.80pt}{0.80pt}
19&es-ar,he-ar,it-ar,pt-ar&ja-ar,ko-ja,tr-ar,fr-ar,pt-fr,it-pt,ko-ar\\\specialrule{0em}{0.80pt}{0.80pt}
20&it-fr,ko-fr,ar-fr,ja-fr&es-fr,pt-fr,tr-fr,ru-fr\\\specialrule{0em}{0.80pt}{0.80pt}
21&pt-ru,ko-ru,fr-ru,it-ru,ar-ru&es-ru,it-pt,he-ru\\\specialrule{0em}{0.80pt}{0.80pt}
22&es-tr,pt-tr,ko-tr,fr-tr,ar-tr&ja-tr,ru-tr\\\specialrule{0em}{0.80pt}{0.80pt}
23&it-ko,tr-ko,ar-ko,fr-ko,es-ko,pt-ko&ja-ko,he-ko,ru-ko\\\specialrule{0em}{0.80pt}{0.80pt}
24&ru-pt,it-pt,es-pt,ar-pt,ko-pt&es-fr,ru-it,pt-it,ja-pt,fr-pt,he-pt\\\specialrule{0em}{0.80pt}{0.80pt}
25&pt-ja,es-ja,tr-ja,ru-ja,fr-ja,ko-ja,it-ja&es-ko,ar-ja\\\specialrule{0em}{0.80pt}{0.80pt}
26&he-it,ko-it,tr-it,fr-it,ar-it,pt-it&fr-es,ko-fr,ru-it,ru-pt,es-it\\\specialrule{0em}{0.80pt}{0.80pt}
""".strip().split()

# differs = []
# for key1 in sorted(affinity_matrix.keys()):
#     for key2 in sorted(affinity_matrix[key1].keys()):
#         if key1 <= key2: continue
#         if affinity_matrix[key1][key2] - affinity_matrix[key2][key1] > 0:
#             differs.append((key1, key2, abs(affinity_matrix[key1][key2] - affinity_matrix[key2][key1])))
#         else:
#             differs.append((key2, key1, abs(affinity_matrix[key1][key2] - affinity_matrix[key2][key1])))
#
# cluster = collections.defaultdict(list)
# for line in res:
#     for t in line.split("\\")[0].split("&")[1].split(","):
#         cluster[t] = line.split("\\")[0].split("&")[2].split(",")
#
# differs = sorted(differs, key=lambda t: -t[-1])
# for key1, key2, d in differs:
#     if key1 not in cluster[key2] and key2 in cluster[key1]:
#         print(key1, key2, d)


differs = []
for key1 in sorted(affinity_matrix.keys()):
    for key2 in sorted(affinity_matrix[key1].keys()):
        if key1 <= key2: continue
        if affinity_matrix[key1][key2] > 0 and affinity_matrix[key2][key1] > 0:
            differs.append((key1, key2, (affinity_matrix[key1][key2] + affinity_matrix[key2][key1])))

cluster = collections.defaultdict(list)
for line in res:
    for t in line.split("\\")[0].split("&")[1].split(","):
        cluster[t] = line.split("\\")[0].split("&")[2].split(",")

differs = sorted(differs, key=lambda t: -t[-1])
for key1, key2, d in differs:
    if key1 in cluster[key2] and key2 in cluster[key1]:
        print(key1, key2, d)


# differs = []
# for key1 in sorted(affinity_matrix.keys()):
#     for key2 in sorted(affinity_matrix[key1].keys()):
#         # if key1 <= key2: continue
#         # if affinity_matrix[key1][key2] < 0 and affinity_matrix[key2][key1] < 0:
#         differs.append((key1, key2, (affinity_matrix[key1][key2] + affinity_matrix[key2][key1])))
#
# cluster = collections.defaultdict(list)
# for line in res:
#     for t in line.split("\\")[0].split("&")[1].split(","):
#         cluster[t] = line.split("\\")[0].split("&")[2].split(",")
#
# differs = sorted(differs, key=lambda t: t[-1])
# for key1, key2, d in differs:
#     if key1 in cluster[key2] and key2 in cluster[key1]:
#         print(key1, key2, d)
