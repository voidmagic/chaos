import sys
import random
import pathlib


num_parallel = 4000

origin_train_path = sys.argv[1]
parallel_train_path = sys.argv[2]

origin_train_path = pathlib.Path(origin_train_path)
parallel_train_path = pathlib.Path(parallel_train_path)
lang_pairs = sys.argv[3].split(',')


for lang_pair in lang_pairs:
    src = lang_pair.split('-')[0]
    tgt = lang_pair.split('-')[1]
    line_set = []
    src_file = parallel_train_path / 'train.{}.{}'.format(lang_pair, src)
    tgt_file = parallel_train_path / 'train.{}.{}'.format(lang_pair, tgt)
    with src_file.open() as f1, tgt_file.open() as f2:
        for line_src, line_tgt in zip(f1, f2):
            line_src = line_src.strip()
            line_tgt = line_tgt.strip()
            line = line_src + '\t' + line_tgt
            line_set.append(line)
    random.seed(0)
    random.shuffle(line_set)
    line_set = line_set[:num_parallel]

    src_result_file = parallel_train_path / 'multi.clean.{}.{}'.format(lang_pair, src)
    tgt_result_file = parallel_train_path / 'multi.clean.{}.{}'.format(lang_pair, tgt)
    with src_result_file.open('w') as f1, tgt_result_file.open('w') as f2:
        for line in line_set:
            line_src = line.split('\t')[0]
            line_tgt = line.split('\t')[1]
            f1.write(line_src + '\n')
            f2.write(line_tgt + '\n')


    line_set = set(line_set)
    src_file = origin_train_path / 'train.{}.{}'.format(lang_pair, src)
    tgt_file = origin_train_path / 'train.{}.{}'.format(lang_pair, tgt)
    src_result_file = parallel_train_path / 'train.clean.{}.{}'.format(lang_pair, src)
    tgt_result_file = parallel_train_path / 'train.clean.{}.{}'.format(lang_pair, tgt)
    with src_file.open() as f1, tgt_file.open() as f2, src_result_file.open('w') as f3, tgt_result_file.open('w') as f4:
        for line_src, line_tgt in zip(f1, f2):
            line_src = line_src.strip()
            line_tgt = line_tgt.strip()
            line = line_src + '\t' + line_tgt
            if line not in line_set:
                f3.write(line_src + '\n')
                f4.write(line_tgt + '\n')
