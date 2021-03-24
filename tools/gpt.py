import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 10)
        # self.make_share()

    def make_share(self):
        self.l1 = self.l2

    def forward(self, x):
        return self.l1(self.l2(x))



model = Model()
state = torch.load('data/a.pt')
model.load_state_dict(state)
print(model.l1.bias)
print(model.l2.bias)
print(model.l1 == model.l2)
# torch.save(model.state_dict(), 'data/a.pt')
model.l1.weight.device