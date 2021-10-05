# usage:
# 1. normalize: spm_normalize --model model file > file.norm
# 2. learn bpe: python train-bpe.py --raw_model model --input=file.norm --vocab_size=2000 --character_coverage=0.99 --model_prefix=bpe
# 3. apply bpe: spm_encode --model bpe.model < file > file.bpe
import multiprocessing
import sys
import os
import argparse
import logging
import collections
import utils.sentencepiece_model_pb2 as model

logging.basicConfig(format="%(asctime)s | %(message)s", level="INFO", stream=sys.stdout)

MAX_SENTENCE_LENGTH = 4192
META_PIECES = ['<unk>', '<s>', '</s>']
K_UNK_BYTE = b"\xe2\x96\x85"
K_UNK_CHAR = K_UNK_BYTE.decode()
K_SPACE_SYMBOL = b"\xe2\x96\x81".decode()
K_MAX_PIECE_LENGTH = 16
K_UPDATE_ACTIVE_SYMBOL_INTERVAL = 100
K_MIN_ACTIVE_SYMBOL_SIZE = 1000
K_TOP_FREQUENT_RATIO = 0.05


class Symbol:
    def __init__(self, left=None, right=None, text=None, is_unk=False, freq=0, positions=None):
        self.left = left
        self.right = right
        self.text = text
        self.is_unk = is_unk
        self.freq = freq
        self.positions = positions or []

    def is_bigram(self):
        return self.left is not None and self.right is not None

    def __str__(self):
        return self.text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--vocab", type=str, help="初始的词表")
    parser.add_argument("--raw_model", type=str, default=os.path.dirname(__file__) + "/utils/model")
    parser.add_argument("--model_prefix", type=str)
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    args = parser.parse_args()
    return args


def process(line_str):
    line_str = line_str.strip()
    line_byte = line_str.encode()
    if len(line_byte) == 0:
        return None
    if len(line_byte) > MAX_SENTENCE_LENGTH:
        return None
    if K_UNK_BYTE in line_byte:
        logging.info("Reserved chars are found. Skipped: " + line_str)
        return None
    return [K_SPACE_SYMBOL + word for word in line_str.split()]


def read_files(input_files):
    char_counter = collections.Counter()
    sentences, raw_sentences = [], []
    for file_idx, filename in enumerate(input_files):
        logging.info("Loading corpus: " + filename)
        with open(filename, encoding='utf-8-sig') as f:
            raw_sentences += f.readlines()

    with multiprocessing.Pool(32) as p:
        sentences = p.map(process, raw_sentences)

    sentences = [s for s in sentences if s is not None]
    for sentence in sentences:
        for word in sentence:
            char_counter.update(word)
    sentences = [s for s in sentences if s is not None]
    logging.info(f"Loaded all {len(sentences)} sentences")
    logging.info(f"Skipped {len(raw_sentences)-len(sentences)} sentences.")
    return sentences, char_counter


def char_set(char_counter, character_coverage, vocab_chars):
    all_char_count = sum(char_counter.values())
    logging.info(f"all chars count={all_char_count}")

    actual_char_count, required_chars = 0, []
    for char, count in sorted(char_counter.items(), key=lambda pair: (-pair[1], pair[0])):
        if char in vocab_chars:
            required_chars.append(char)
            actual_char_count += count
    logging.info(f"Alphabet size={len(required_chars)}")
    logging.info(f"Done: {actual_char_count / all_char_count :2.4%} characters are covered.")
    return required_chars


def word_set(sentences, required_chars):
    required_chars = set(required_chars)
    word_counter = collections.Counter()
    for sentence in sentences:
        for word in sentence:
            new_word = "".join([c if c in required_chars else K_UNK_CHAR for c in word])
            word_counter.update([new_word])
    word_counter = collections.OrderedDict(sorted(word_counter.items(), key=lambda pair: (-pair[1], pair[0])))
    logging.info(f"Done! {len(word_counter)} (number of words)")
    return word_counter, list(word_counter.items())


def get_char_symbol(symbol_cache, chars, freq=None):
    if chars not in symbol_cache:
        symbol_cache[chars] = Symbol(text=chars, is_unk=chars == K_UNK_CHAR, freq=freq)
    return symbol_cache[chars]


def initialize_symbols(symbol_cache, word_counter, char_counter):
    # Initializes symbols. symbols[i][j] stores an unary symbol.
    symbols = [[] for _ in word_counter.items()]
    for idx, (word, _) in enumerate(word_counter.items()):
        symbols[idx] = [get_char_symbol(symbol_cache, c, freq=char_counter[c]) for c in word]
    return symbols


def get_pair_symbol(symbol_cache, left: Symbol, right: Symbol):
    if left is None or right is None or left.is_unk or right.is_unk:
        return None
    chars = left.text + right.text
    if len(chars) > K_MAX_PIECE_LENGTH:
        return
    if chars not in symbol_cache:
        symbol_cache[chars] = Symbol(left=left, right=right, text=chars)
    return symbol_cache[chars]


def add_new_pair(symbol_cache, word_symbols, active_symbols, sid, left, right):
    if left == -1 or right == -1: return
    symbol = get_pair_symbol(symbol_cache, word_symbols[sid][left], word_symbols[sid][right])
    if symbol is not None:
        active_symbols.add(symbol)
        symbol.positions.append((sid, left, right))


def initialize_bigram_symbols(symbol_cache, symbols):
    active_symbols = set()
    for sid, symbol in enumerate(symbols):
        for i in range(1, len(symbol)):
            add_new_pair(symbol_cache, symbols, active_symbols, sid, i - 1, i)
    return active_symbols


def compute_frequency(symbol: Symbol, word_symbols, word_counter):
    if symbol.freq > 0: return
    prev_sid, prev_left, prev_right, idx = -1, 0, 0, 0
    while idx < len(symbol.positions):
        sid, left, right = symbol.positions[idx]
        if (prev_sid == sid and left == prev_right) or symbol.left != word_symbols[sid][left] or symbol.right != word_symbols[sid][right]:
            del symbol.positions[idx]
            prev_sid, prev_left, prev_right = -1, 0, 0
        else:
            symbol.freq += word_counter[sid][1]
            prev_sid, prev_left, prev_right = sid, left, right
            idx += 1


def update_active_symbols(symbol_cache, active_symbols, word_symbols, word_counter_list, require_all):
    if require_all:
        active_symbols.clear()
        active_symbols.update([symbol for symbol in symbol_cache.values() if symbol.is_bigram()])
        return

    symbols = []
    for symbol in symbol_cache.values():
        if symbol.is_bigram():
            compute_frequency(symbol, word_symbols, word_counter_list)
            symbols.append(symbol)

    size = int(min(max(K_MIN_ACTIVE_SYMBOL_SIZE, len(symbol_cache) * K_TOP_FREQUENT_RATIO), len(symbols)))
    symbols = sorted(symbols, key=lambda s: s.freq, reverse=True)[:size]
    logging.info(f"Updating active symbols. max_freq={symbols[0].freq} min_freq={symbols[-1].freq}")
    active_symbols.clear()
    active_symbols.update(symbols)


def loop(vocab_size, symbol_cache, active_symbols, word_symbols, word_counter_list, vocab_words):
    final_pieces = list()
    while len(final_pieces) < vocab_size:
        if len(final_pieces) % K_UPDATE_ACTIVE_SYMBOL_INTERVAL == 0:
            update_active_symbols(symbol_cache, active_symbols, word_symbols, word_counter_list, require_all=len(vocab_words)>0)

        best_symbol: Symbol = None

        if len(vocab_words) > 0:
            first_word = vocab_words.pop()
            best_symbol = symbol_cache.get(first_word)
            if best_symbol is None:
                final_pieces.append(first_word)
                continue
        else:
            for symbol in active_symbols:
                compute_frequency(symbol, word_symbols, word_counter_list)
                if best_symbol is None or (symbol.freq > best_symbol.freq or (
                        symbol.freq == best_symbol.freq and (len(symbol.text) < len(best_symbol.text) or (
                        len(symbol.text) == len(best_symbol.text) and symbol.text < best_symbol.text)))):
                    best_symbol = symbol

        if best_symbol is None:
            logging.info("No valid symbol found")
            break

        if best_symbol.text not in final_pieces:
            final_pieces.append(best_symbol.text)
            logging.info(f"Added: freq={best_symbol.freq} size={len(final_pieces)} all={len(symbol_cache)} active={len(active_symbols)} piece={best_symbol.text}")

        for sid, left, right in best_symbol.positions:
            if word_symbols[sid][left] is None:
                continue

            def get_next_index():
                for i in range(right + 1, len(word_symbols[sid])):
                    if word_symbols[sid][i] is None: continue
                    return i
                return -1

            def get_prev_index():
                for i in range(left - 1, -1, -1):
                    if word_symbols[sid][i] is None: continue
                    return i
                return -1

            next_idx = get_next_index()
            prev_idx = get_prev_index()

            def reset_freq(left_, right_):
                if left == -1 or right == -1: return
                symbol_ = get_pair_symbol(symbol_cache, word_symbols[sid][left_], word_symbols[sid][right_])
                if symbol_ is not None and symbol_ != best_symbol:
                    symbol_.freq = 0

            reset_freq(prev_idx, left)
            reset_freq(right, next_idx)

            word_symbols[sid][left] = best_symbol
            word_symbols[sid][right] = None

            add_new_pair(symbol_cache, word_symbols, active_symbols, sid, prev_idx, left)
            add_new_pair(symbol_cache, word_symbols, active_symbols, sid, left, next_idx)

        del symbol_cache[best_symbol.text]
        active_symbols.remove(best_symbol)

    return final_pieces


def save(pieces, model_prefix, raw_model):
    model_proto = model.ModelProto()
    model_proto.ParseFromString(open(raw_model, "rb").read())

    for piece in model_proto.pieces[3:]:
        model_proto.pieces.remove(piece)

    for idx, piece in enumerate(pieces):
        new_token = model.ModelProto().SentencePiece()
        new_token.piece = piece
        new_token.score = -idx
        model_proto.pieces.append(new_token)

    logging.info(f"Saving model: {model_prefix}.model")
    with open(model_prefix + ".model", "wb") as f:
        f.write(model_proto.SerializeToString())

    logging.info(f"Saving vocabs: {model_prefix}.vocab")
    with open(model_prefix + ".vocab", "w") as f:
        for token in model_proto.pieces:
            f.write(f"{token.piece}\t{int(token.score)}\n")


def train():
    args = parse_args()

    # load initial vocab
    with open(args.vocab) as f:
        vocab = [l.strip().split()[0] for l in f][3:]
        index = vocab.index(K_SPACE_SYMBOL)
        vocab_words = vocab[:index]
        vocab_chars = vocab[index:]
        vocab_words.reverse()
    logging.info(f"Initial words {len(vocab_words)} chars {len(vocab_chars)}")
    sentences, char_counter = read_files(args.input.split(','))
    required_chars = vocab_chars
    word_counter, word_counter_list = word_set(sentences, required_chars)

    symbol_cache = dict()  # text to symbol
    word_symbols = initialize_symbols(symbol_cache, word_counter, char_counter)
    active_symbols = initialize_bigram_symbols(symbol_cache, word_symbols)
    vocab_size = args.vocab_size - len(META_PIECES) - len(required_chars)

    if vocab_size < len(vocab_words):
        logging.info(f"Vocab size too small {vocab_size} < {len(vocab_words)}")
        return

    final_pieces = loop(vocab_size, symbol_cache, active_symbols, word_symbols, word_counter_list, vocab_words)

    save(final_pieces + required_chars, args.model_prefix, args.raw_model)


if __name__ == '__main__':
    train()
