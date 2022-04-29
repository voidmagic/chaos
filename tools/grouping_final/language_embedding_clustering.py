import collections

import torch
import re
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from matplotlib import pyplot as plt
import numpy as np


root_path = "/home/qwang/028-cluster/003-models/601-multi-tedx/m2o"
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


def cluster_n(n_clusters):
    clusters = cut_tree(Z, n_clusters=n_clusters)
    langs = list(key_embedding.keys())

    language_clusters = collections.defaultdict(list)
    for cls, lang in zip(clusters.T[0], langs):
        language_clusters[cls].append(lang)

    score = []
    for key, value in language_clusters.items():
        print(key, value)

        tensors = [key_embedding[lang] for lang in value]
        differs = sum([(t - np.mean(tensors, axis=0)) ** 2 for t in tensors])
        score.append(differs)
    elbow_score = np.sum(score)

    return elbow_score

sse = {}
for i in range(1, len(key_embedding) + 1):
    sse[i] = cluster_n(i)

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
