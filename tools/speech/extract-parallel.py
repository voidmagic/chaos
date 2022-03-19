

langs = "de es fr it nl pt ro ru".split()

sets = []

for lang in langs:
    filename = "/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/tst-COMMON_{}_stt.tsv".format(lang)
    with open(filename) as f:
        f.readline()
        id_set = set()
        for line in f.readlines():
            ids = line.strip().split("\t")[0]
            id_set.add(ids)
    sets.append(id_set)

s1 = sets[0]
for s in sets[1:]:
    s1 = s1 & s

for lang in langs:
    filename = "/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/tst-COMMON_{}_stt.tsv".format(lang)
    out_filename = "/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/aligntst_{}_stt.tsv".format(lang)
    with open(filename) as f, open(out_filename, "w") as f1:
        head = f.readline()
        f1.write(head)
        for line in f.readlines():
            ids = line.strip().split("\t")[0]
            if ids in s1:
                f1.write(line)
