import collections
import math
import numpy as np

bilingual_prop_path = "/home/qwang/030-transfer/003-models/411-bilingual-iwslt/en-zh/zh.prop"
eszh_prop_path = "/home/qwang/030-transfer/003-models/410-multilingual-iwslt/eszh-o2m/zh.prop"


def parse_file(filename):
    partial_dict = collections.defaultdict(dict)
    with open(filename) as f:
        for line in f.readlines():
            line = line.strip()
            sid = int(line.split("\t")[0].split('-')[1])
            if line.startswith("S"):
                source = line.split("\t")[1]
                partial_dict[sid]["source"] = source
            elif line.startswith("T"):
                target = line.split("\t")[1]
                target = target.split()
                target = [word[10:] for word in target[:-1]]
                target = " ".join(target)
                partial_dict[sid]["target"] = target
            elif line.startswith("L"):
                loss = line.split("\t")[1]
                loss = loss.split()
                loss = loss[:-1]
                loss = " ".join(loss)
                partial_dict[sid]["loss"] = loss
    return partial_dict


bi_dict = parse_file(bilingual_prop_path)
eszh_dict = parse_file(eszh_prop_path)

bi_loss = [float(v) for value in bi_dict.values() for v in value["loss"].split()]
eszh_loss = [float(v) for value in eszh_dict.values() for v in value["loss"].split()]

ideal_loss = [sum(bi_loss) / len(bi_loss)] * len(bi_loss)


print("bilin loss\t", -sum([math.log(v+1e-7) for v in bi_loss]) / len(bi_loss))
print("eszh loss\t ", -sum([math.log(v+1e-7) for v in eszh_loss]) / len(bi_loss))

print()

print("bilin mean prop\t", sum(bi_loss) / len(bi_loss))
print("eszh mean prop\t ", sum(eszh_loss) / len(bi_loss))

print()


print("ideal bilin loss\t", -math.log(sum(bi_loss) / len(bi_loss)))
print("ideal eszh loss \t", -math.log(sum(eszh_loss) / len(bi_loss)))


print()

print("var bilin\t", np.var(bi_loss))
print("var eszh \t", np.var(eszh_loss))

print()

combine_dict = {
    key: {
        "source": bi_dict[key]["source"],
        "target": bi_dict[key]["target"].split()[:-1],
        "losses": {
            "bi": [float(v) for v in bi_dict[key]["loss"].split()],
            "eszh": [float(v) for v in eszh_dict[key]["loss"].split()],
        },
    }
    for key in bi_dict.keys()
}

counter = collections.defaultdict(int)

for key, value in combine_dict.items():
    for word, bil, eszhl in zip(value['target'], value['losses']['bi'], value['losses']['eszh']):
        if eszhl > bil:
            counter[word] += 1
        elif bil > eszhl:
            counter[word] -= 1


dictionary = collections.Counter()
# train_path = "/home/qwang/030-transfer/002-dataset/301-iwslt-bpe/train.en-zh.zh"
# with open(train_path) as f:
#     for line in f.readlines():
#         line = line.strip().split()
#         dictionary.update(line)

# train_path = "/home/qwang/030-transfer/002-dataset/301-iwslt-bpe/train.en-es.es"
# with open(train_path) as f:
#     for line in f.readlines():
#         line = line.strip().split()
#         dictionary.update(line)

# s1, s2 = [], []
# for key, value in sorted(counter.items(), key=lambda p: p[1]):
#     if value > 0:
#         s1.append(dictionary[key])
#     else:
#         s2.append(dictionary[key])
#
# print(np.mean(s1), np.mean(s2))


s = []
for key, value in sorted(counter.items(), key=lambda p: p[1]):
    if value > 20:
        s.append(key)
print("\n".join(sorted(s)))
