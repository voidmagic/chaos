import math
import sys
import collections


counter = collections.Counter()
for line in sys.stdin:
    line = line.strip().split()
    counter.update(line)

total_word = sum(counter.values())
entropy = [-(freq / total_word) * math.log(freq / total_word, 2) for word, freq in counter.items()]
entropy = round(sum(entropy), 4)
print(entropy)
