import csv
import gzip
import sys


gz_path = sys.argv[1]
output_dir = sys.argv[2]

lines = []
with gzip.open(gz_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    target_langs = reader.fieldnames[2:]
    file_dict = {lang: (open(f"{output_dir}/en-{lang.replace('-', '')}.en", "w"), open(f"{output_dir}/en-{lang.replace('-', '')}.{lang.replace('-', '')}", "w")) for lang in target_langs}
    for line in reader:
        if len(line['en']) == 0: continue
        for lang in target_langs:
            if len(line[lang]) == 0: continue
            file_dict[lang][0].write(line['en'] + "\n")
            file_dict[lang][1].write(line[lang] + "\n")

    for lang in target_langs:
        file_dict[lang][0].close()
        file_dict[lang][1].close()
