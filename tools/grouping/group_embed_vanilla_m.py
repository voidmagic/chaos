import collections

import torch
import re
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import numpy as np


root_path = "/home/qwang/027-optim/003-models/622-aftm-tedx-base/m2o"
model_path = f"{root_path}/checkpoint_average.pt"
log_path = f"{root_path}/log.txt"

model = torch.load(model_path)
embedding = model["model"]["encoder.embed_tokens.weight"]

key_embedding = {}
sorted_langs = []
with open(log_path) as f:
    for line in f:
        if "fairseq.tasks.multilingual_translation" not in line or "dictionary" not in line:
            continue
        key = re.findall(r".*\[(.*)\].*", line)[0]
        sorted_langs.append(key)

for lang, emb in zip(sorted_langs, embedding[-len(sorted_langs):]):
    key_embedding[lang] = emb.tolist()

del key_embedding["en"]
Z = linkage(np.array(list(key_embedding.values())), 'complete', metric="cosine")
fig = plt.figure(figsize=(15, 10))
dn = dendrogram(Z, labels=np.array(list(key_embedding.keys())))
plt.show()

clusters = cut_tree(Z, n_clusters=7)
langs = list(key_embedding.keys())

language_clusters = collections.defaultdict(list)
for cls, lang in zip(clusters.T[0], langs):
    language_clusters[cls].append(lang)

for key, value in language_clusters.items():
    print(key, value)
