import csv
import gzip
import sys


gz_path = sys.argv[1]
output_dir = sys.argv[2]

lines = []
with gzip.open(gz_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    languages = "es fr ar ko ru tr it ja he pt".split()
    file_dict = {}
    for lang1 in languages:
        for lang2 in languages:
            if lang1 == lang2:
                continue
            file_dict[f"{lang1}-{lang2}"] = {
                lang1: open(f"{output_dir}/{lang1}-{lang2}.{lang1}", "w"),
                lang2: open(f"{output_dir}/{lang1}-{lang2}.{lang2}", "w")
            }

    for line in reader:
        for lang1 in languages:
            if len(line[lang1]) == 0:
                continue
            for lang2 in languages:
                if len(line[lang2]) == 0 or lang1 == lang2:
                    continue
                file_dict[f"{lang1}-{lang2}"][lang1].write(line[lang1] + "\n")
                file_dict[f"{lang1}-{lang2}"][lang2].write(line[lang2] + "\n")

    for lang1 in languages:
        for lang2 in languages:
            if lang1 == lang2:
                continue
            file_dict[f"{lang1}-{lang2}"][lang2].close()
            file_dict[f"{lang1}-{lang2}"][lang2].close()
