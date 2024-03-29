import torch
from sklearn.metrics.pairwise import cosine_similarity

folder = "/mnt/hdd/qwang/029-must/004-analyze/001-gradients"

langs = "de es fr it nl pt ro ru".split()

prefix = "decoder.layers.5"

a = []
for lang1 in langs:
    b = []
    for lang2 in langs:
        gradients1 = torch.load(f"{folder}/{lang1}.pt")
        gradients2 = torch.load(f"{folder}/{lang2}.pt")
        grads1 = []
        grads2 = []
        for key in gradients1.keys():
            # print(key)
            if key.startswith(prefix):
                grads1.append(gradients1[key])
                grads2.append(gradients2[key])
        b.append(cosine_similarity(torch.cat(grads1).view(1, -1), torch.cat(grads2).view(1, -1))[0][0])
    a.append(b)

for i in range(len(a)):
    for j in range(len(a[0])):
        print(a[j][i], end="\t")
    print()
