import argparse
import collections
import multiprocessing
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-k", type=int, default=1, help="top-k")
    parser.add_argument("-l", type=int, default=0, help="最短词长度")
    args = parser.parse_args()
    return args


def read_vocab(path):
    vocab = list()
    with open(os.path.join(path, "bpe.vocab")) as f:
        for line in f.readlines()[3:]:
            line = line.strip().split()
            vocab.append(line[0])
    return vocab


def get_langs(path):
    langs = set()
    for f in os.listdir(path):
        if f.startswith('train'):
            f = f.split('.')[-1]
            langs.add(f)
    return sorted(list(langs))


def count_file(fns):
    lang, fn = fns
    counter = collections.Counter()
    with open(fn) as f:
        for line in f.readlines():
            line = line.strip().split()
            counter.update(line)
    return lang, counter


def count_files(langs, path):
    fns = []
    for lang in langs:
        for fn in os.listdir(path):
            if fn.endswith(lang) and fn.startswith('train'):
                fns.append((lang, os.path.join(path, fn)))
    with multiprocessing.Pool(len(fns)) as pool:
        counters = pool.map(count_file, fns)

    lang_counter = collections.defaultdict(collections.Counter)
    for lang, counter in counters:
        lang_counter[lang].update(counter)
    return lang_counter


def word_for_lang(vocab, counter, k, l):
    result = dict()
    for word in vocab:
        if len(word) <= l: continue
        pairs = [(lang, ctr[word]) for lang, ctr in counter.items() if word in ctr and ctr[word] > 0]
        pairs = sorted(pairs, key=lambda p: -p[1])[:k]
        result[word] = [p[0] for p in pairs]
    return result


def main():
    args = parse_args()
    path = args.path
    vocab = read_vocab(path)
    langs = get_langs(path)
    counter = count_files(langs, path)
    word_lang = word_for_lang(vocab, counter, args.k, args.l)
    ctr = collections.Counter()
    for _langs in word_lang.values():
        ctr.update(_langs)

    for lang, freq in sorted(ctr.items()):
        print(lang, '\t', freq)


if __name__ == '__main__':
    main()
