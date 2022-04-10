import collections

import torch
import re
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import numpy as np


root_path = "/home/qwang/027-optim/003-models/601-multi-tedx/o2m"
model_path = f"{root_path}/checkpoint_average.pt"
log_path = f"{root_path}/log.txt"

model = torch.load(model_path)
embedding = model["model"]["encoder.embed_tokens.weight"]

key_embedding = {}
with open(log_path) as f:
    for line in f:
        if "fairseq.data.multilingual.multilingual_data_manager" not in line or "src_langtok" not in line:
            continue
        key = re.findall(r".*main:(.*) src.*", line)[0]
        index = re.findall(r".*src_langtok: (.*);.*", line)[0]
        key_embedding[key] = embedding[int(index)].tolist()

Z = linkage(np.array(list(key_embedding.values())), 'complete', metric="cosine")
fig = plt.figure(figsize=(15, 10))
dn = dendrogram(Z, labels=np.array(list(key_embedding.keys())))
plt.show()

clusters = cut_tree(Z, n_clusters=6)
langs = list(key_embedding.keys())

language_clusters = collections.defaultdict(list)
for cls, lang in zip(clusters.T[0], langs):
    language_clusters[cls].append(lang)

for key, value in language_clusters.items():
    print(key, value)
