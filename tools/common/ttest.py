import numpy as np
from scipy.stats import ttest_ind
import sys


sys1 = sys.argv[1]
sys2 = sys.argv[2]

score1 = []
with open(sys1) as f:
    for line in f:
        score1.append(float(line.strip()))

score2 = []
with open(sys2) as f:
    for line in f:
        score2.append(float(line.strip()))

res = ttest_ind(score1, score2)

print(res)
