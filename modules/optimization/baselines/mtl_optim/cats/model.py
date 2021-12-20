import torch
import torch.nn as nn
from torch import linalg as LA

class ModelParetoWeightLambda(nn.Module):
    def __init__(self, n):
        super(ModelParetoWeightLambda, self).__init__()
        self.alpha = nn.Parameter(torch.ones(n))
        self.lambda_all = nn.Parameter(torch.zeros(n))
        self.lambda_c = nn.Parameter(torch.zeros(1))
        self.lambda_s = nn.Parameter(torch.zeros(1))
        self.eps = torch.tensor(0.001)

    def forward(self, loss_vec, grads):
        part_1 = torch.dot(self.alpha, loss_vec)
        part_2 = self.lambda_c * (self.eps - LA.vector_norm(self.alpha.view(-1, 1) * grads) ** 2)
        part_3 = self.lambda_s * (torch.sum(self.alpha) - 1) ** 2
        part_4 = torch.dot(self.lambda_all, (self.alpha - self.eps))
        return part_1 - part_2 + part_3 - part_4