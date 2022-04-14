"""
pip install mecab-python3
pip install unidic-lite
pip install jieba
pip install sacremoses
python -m pip install python-dev-tools
pip install -v python-mecab-ko
"""

import argparse
import jieba
import sys
import MeCab
from sacremoses import MosesTokenizer
import mecab


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', default='en', type=str, help="language")
args = parser.parse_args()
mecab = mecab.MeCab()

lang = args.lang

jieba.setLogLevel('ERROR')
tagger = MeCab.Tagger("-Owakati")
mt = MosesTokenizer(lang=lang)

for line in sys.stdin:
    if lang == 'zh':
        line = ' '.join(jieba.cut(line)).strip()
    elif lang == 'ja':
        line = tagger.parse(line).strip()
    elif lang == 'ko':
        line = " ".join(mecab.morphs(line.strip()))
    else:
        line = mt.tokenize(line, return_str=True).strip()
    print(line)
