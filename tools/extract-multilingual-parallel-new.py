import argparse
import os
import random


def process(split, lang_pairs, path, output):
    source_hub = len(set([langs[0] for langs in lang_pairs])) == 1
    if not os.path.exists(output):
        os.mkdir(output)

    sent_pairs = {}
    for src, tgt in lang_pairs:
        src_f = os.path.join(path, '{}.{}-{}.{}'.format(split, src, tgt, src))
        tgt_f = os.path.join(path, '{}.{}-{}.{}'.format(split, src, tgt, tgt))

        with open(src_f) as src_f, open(tgt_f) as tgt_f:
            sent_pairs["{} {}".format(src, tgt)] = dict(zip(src_f, tgt_f)) if source_hub else dict(zip(tgt_f, src_f))

    hub_sentences = [value.keys() for value in sent_pairs.values()]
    common_lines = list(set(hub_sentences[0]).intersection(*map(set, hub_sentences[1:])))
    random.shuffle(common_lines)
    print(split + ":", len(common_lines))

    for src, tgt in lang_pairs:
        src_f = os.path.join(output, '{}.{}-{}.{}'.format(split, src, tgt, src))
        tgt_f = os.path.join(output, '{}.{}-{}.{}'.format(split, src, tgt, tgt))
        with open(src_f, 'w') as src_f, open(tgt_f, 'w') as tgt_f:
            for hub_line in common_lines:
                src_line = hub_line if source_hub else sent_pairs["{} {}".format(src, tgt)][hub_line]
                tgt_line = sent_pairs["{} {}".format(src, tgt)][hub_line] if source_hub else hub_line
                src_f.write(src_line)
                tgt_f.write(tgt_line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--lang-pairs")
    parser.add_argument("--output-dir")
    parser.add_argument("--part", default='train,valid,test')
    args = parser.parse_args()
    lang_pairs = args.lang_pairs.split(',')
    lang_pairs = [lang_pair.split('-') for lang_pair in lang_pairs]
    for split in args.part.split(','):
        process(split, lang_pairs, args.data, args.output_dir)


if __name__ == '__main__':
    main()
