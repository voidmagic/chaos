import math
import sys
import collections

norm = len(sys.argv) == 2 and sys.argv[1] == 'norm'

counter = collections.Counter()
for line in sys.stdin:
    line = line.strip().split()
    counter.update(line)

total_word = sum(counter.values())
if norm:
    entropy = [- 1 / len(word) * (freq / total_word) * math.log(freq / total_word, 2) for word, freq in counter.items()]
else:
    entropy = [- 1 * (freq / total_word) * math.log(freq / total_word, 2) for word, freq in counter.items()]
entropy = round(sum(entropy), 4)
print(entropy)
