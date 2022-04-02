import math

import tqdm
import numpy as np

root_path = "/home/qwang/027-optim/003-models/210-multilingual-ted/diverse/m2o/proportional"
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


num_steps, step_start, step_end = 0, 0, math.inf
diff_sum = 0
with open(log_path) as f:
    for line in tqdm.tqdm(f.readlines()):
        if "Affinity" not in line or "modules.optimization.affinity.affinity_task" not in line:
            continue
        num_steps += 1
        if step_start < num_steps < step_end:
            languages = line.strip().split(" | ")[4].strip("['] ").split("', '")
            languages = [lang.strip('_') for lang in languages]
            target = line.strip().split(" | ")[6][start:end]
            loss_before = float(line.strip().split(" | ")[7])
            loss_after = float(line.strip().split(" | ")[8])
            loss_differ = float(line.strip().split(" | ")[9])
            diff_sum += (loss_before - loss_after)
            for lang in languages:
                affinity_matrix[target][lang].append((loss_before - loss_after) / len(languages))
print(diff_sum)
print("\t" + "\t".join(sorted(affinity_matrix.keys())))
for key1 in sorted(affinity_matrix.keys()):
    print(key1, end="\t")
    for key2 in sorted(affinity_matrix[key1].keys()):
        affinity_matrix[key1][key2] = np.sum(affinity_matrix[key1][key2] or [0])
        print("%.8f" % float(affinity_matrix[key1][key2]), end="\t")
    print()



