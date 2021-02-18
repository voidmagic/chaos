import argparse
import os
import random


def find_multi_parallel(sent_pairs):

    # 去重
    for key, value in sent_pairs.items():
        sentence_pairs = []
        for pair in sorted(value):
            if len(sentence_pairs) == 0:
                sentence_pairs.append(pair)
            if pair[0] != sentence_pairs[-1][0]:
                sentence_pairs.append(pair)
        sent_pairs[key] = sentence_pairs

    sent_pairs_flat = []
    for key, value in sent_pairs.items():
        src, tgt = key.split()
        for sent_pair in value:
            sent_pairs_flat.append(list(sent_pair) + [src, tgt])
    common_lines = []
    common_line = {}
    for src_line, tgt_line, src, tgt in sorted(sent_pairs_flat, key=lambda p: p[0]):
        if not common_line:
            common_line = {src: src_line, tgt: tgt_line}
        elif common_line[src] == src_line:
            common_line[tgt] = tgt_line
        elif len(common_line) == len(sent_pairs.keys()) + 1:
            common_lines.append(common_line)
            common_line = {src: src_line, tgt: tgt_line}
        else:
            common_line = {src: src_line, tgt: tgt_line}
    random.shuffle(common_lines)
    return common_lines


def process(split, lang_pairs, path, output):
    assert len(set([langs[0] for langs in lang_pairs])) == 1
    if not os.path.exists(output):
        os.mkdir(output)

    sent_pairs = {}
    for src, tgt in lang_pairs:
        src_f = os.path.join(path, '{}.{}-{}.{}'.format(split, src, tgt, src))
        tgt_f = os.path.join(path, '{}.{}-{}.{}'.format(split, src, tgt, tgt))

        with open(src_f) as src_f, open(tgt_f) as tgt_f:
            sent_pairs["{} {}".format(src, tgt)] = list(zip(src_f, tgt_f))

    common_lines = find_multi_parallel(sent_pairs)
    print(split + ":", len(common_lines))

    for src, tgt in lang_pairs:
        src_f = os.path.join(output, '{}.{}-{}.{}'.format(split, src, tgt, src))
        tgt_f = os.path.join(output, '{}.{}-{}.{}'.format(split, src, tgt, tgt))
        with open(src_f, 'w') as src_f, open(tgt_f, 'w') as tgt_f:
            for lines in common_lines:
                src_f.write(lines[src])
                tgt_f.write(lines[tgt])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--lang-pairs")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    lang_pairs = args.lang_pairs.split(',')
    lang_pairs = [lang_pair.split('-') for lang_pair in lang_pairs]
    for split in ['train', 'valid', 'test']:
        process(split, lang_pairs, args.data, args.output_dir)


if __name__ == '__main__':
    main()
