"""
Allocating Large Vocabulary Capacity for Cross-lingual Language Model Pre-training
"""

import math
import sys
import collections

counter = collections.Counter()
sentences = []
for line in sys.stdin:
    line = line.strip().split()
    counter.update(line)
    sentences.append(line)

total_word = sum(counter.values())

for key, value in counter.items():
    counter[key] = math.log(value / total_word, 2)

log_p = []
for sent in sentences:
    log_p.append(sum([counter[w] for w in sent]))
alp = round(sum(log_p) / len(log_p), 4)
print(alp)
