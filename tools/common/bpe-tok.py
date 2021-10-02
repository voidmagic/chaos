import sentencepiece as spm
from pathlib import Path
import sys
model_file = Path(__file__).parent / 'sentence.bpe.model'
sp = spm.SentencePieceProcessor(model_file=str(model_file))


def encode(line):
    return " ".join(sp.encode(line.strip(), out_type=str))


def decode(line):
    return sp.decode(line.strip().split())


assert len(sys.argv) == 2 and sys.argv[1] in ['-e', '-d']

if sys.argv[1] == '-e':
    f = encode
else:
    f = decode


for ll in sys.stdin:
    print(f(ll))

