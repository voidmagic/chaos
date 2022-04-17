import math

import tqdm
import numpy as np

root_path = "/home/qwang/027-optim/003-models/623-aft-tedx-base-2/o2m"
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
        if "Affinity" not in line or "modules.preprocessing.affinity.affinity_task" not in line:
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



