import collections

import torch
import re
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import numpy as np
import ast

root_path = "/home/qwang/027-optim/003-models/621-emb-tedx/o2m-m1.0-l5.0"
model_path = f"{root_path}/checkpoint_average.pt"
log_path = f"{root_path}/log.txt"

model = torch.load(model_path)
embedding = model["model"]["encoder.weight_s"].T.tolist()

languages = []
with open(log_path) as f:
    for line in f:
        if "parsed the language list as they are ordered in the option" not in line:
            continue
        index = re.findall(r".*: (.*)", line)[0]
        index = ast.literal_eval(index)
        languages += index

key_embedding = dict(list(zip(languages, embedding))[1:])

Z = linkage(np.array(list(key_embedding.values())), 'complete', metric="cosine")
fig = plt.figure(figsize=(15, 10))
dn = dendrogram(Z, labels=np.array(list(key_embedding.keys())))
plt.show()

clusters = cut_tree(Z, n_clusters=4)
langs = list(key_embedding.keys())

language_clusters = collections.defaultdict(list)
for cls, lang in zip(clusters.T[0], langs):
    language_clusters[cls].append(lang)

for key, value in language_clusters.items():
    print(key, value)


for lang, emb in key_embedding.items():
    print(lang, end="\t")
    for e in emb:
        print(e, end="\t")
    print()
