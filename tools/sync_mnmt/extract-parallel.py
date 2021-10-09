import multiprocessing
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str, help="e.g., /data/train")
    parser.add_argument('--lang-pairs', '-l', type=str)
    parser.add_argument('--output', '-o', type=str)
    args = parser.parse_args()
    assert args.prefix is not None
    assert args.lang_pairs is not None
    assert args.output is not None
    return args


def read_file(filename):
    with open(filename[0]) as f0, open(filename[1]) as f1:
        sentences = {src.strip(): tgt.strip() for src, tgt in zip(f0, f1)}
    return sentences, set([sentence for sentence in sentences.keys()])


def read_files(prefix, lang_pairs):
    lang_pairs = [pair.split('-') for pair in lang_pairs.split(',')]

    source_count, target_count = len(set(pair[0] for pair in lang_pairs)), len(set(pair[1] for pair in lang_pairs))
    assert (source_count == 1) != (target_count == 1)

    filenames = [
        (f"{prefix}.{source}-{target}.{source}", f"{prefix}.{source}-{target}.{target}") for source, target in lang_pairs
    ] if source_count == 1 else [
        (f"{prefix}.{source}-{target}.{target}", f"{prefix}.{source}-{target}.{source}") for source, target in lang_pairs
    ]

    with multiprocessing.Pool(len(filenames)) as pool:
        sentences, source_sets = zip(*pool.map(read_file, filenames))
    unique_source = source_sets[0].intersection(*source_sets)
    print(f'Find multilingual parallel {len(unique_source)}')
    return [
        [(source, sentence_pair[source]) for source in unique_source] for sentence_pair in sentences
    ] if source_count == 1 else [
        [(sentence_pair[source], source) for source in unique_source] for sentence_pair in sentences
    ]


def write_file(name_and_content):
    filename, sentences = name_and_content
    with open(filename[0], 'w') as f0, open(filename[1], 'w') as f1:
        for src, tgt in sentences:
            f0.write(src + '\n')
            f1.write(tgt + '\n')


def write_files(sentences, output, lang_pairs):
    lang_pairs = [pair.split('-') for pair in lang_pairs.split(',')]
    filenames = [
        (f"{output}.{source}-{target}.{source}", f"{output}.{source}-{target}.{target}") for source, target in lang_pairs
    ]
    with multiprocessing.Pool(len(filenames)) as pool:
        pool.map(write_file, zip(filenames, sentences))


def main():
    args = parse_args()
    sentences = read_files(args.prefix, args.lang_pairs)
    write_files(sentences, args.output, args.lang_pairs)


if __name__ == '__main__':
    main()

