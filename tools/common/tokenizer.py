"""
pip install mecab-python3
pip install unidic-lite
pip install jieba
pip install sacremoses
"""

import argparse
import jieba
import sys
import MeCab
from sacremoses import MosesTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--lang', default='en', type=str, help="language")
args = parser.parse_args()

lang = args.lang

jieba.setLogLevel('ERROR')
tagger = MeCab.Tagger("-Owakati")
ko_tag = MeCab.Tagger()
mt = MosesTokenizer(lang=lang)

for line in sys.stdin:
    if lang == 'zh':
        line = ' '.join(jieba.cut(line)).strip()
    elif lang == 'ja':
        line = tagger.parse(line).strip()
    else:
        line = mt.tokenize(line, return_str=True).strip()
    print(line)
