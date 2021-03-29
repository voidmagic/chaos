import torch
import torch.nn as nn


class Small(nn.Module):
    def __init__(self):
        super(Small, self).__init__()
        self.linear = nn.Linear(10, 10)


class ModelA(nn.Module):
    def __init__(self):
        super(ModelA, self).__init__()
        self.s1 = Small()
        self.s2 = Small()

        self.s1.linear = self.s2.linear


ma = ModelA()
stat = torch.load('data/m.pt')
print(ma.s1.linear.bias)
print(ma.s2.linear.bias)
ma.s1.load_state_dict(stat)

print(ma.s1.linear.bias)
print(ma.s2.linear.bias)
