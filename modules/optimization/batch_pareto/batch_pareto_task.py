from dataclasses import dataclass, field

import torch
import torch.nn.functional as func
from torch.nn.utils import parameters_to_vector
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, TranslationConfig


@dataclass
class BatchParetoConfig(TranslationConfig):
    fuse_manner: str = field(
        default='none',
        metadata={
            "help": "source language",
        },
    )


def _min_norm_element_from2(v1v1: torch.Tensor, v1v2: torch.Tensor, v2v2: torch.Tensor):
    r"""
    Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
    d is the distance (objective) optimized
    v1v1 = <x1,x1>
    v1v2 = <x1,x2>
    v2v2 = <x2,x2>
    """
    if v1v2 >= v1v1:
        # Case: Fig 1, third column
        gamma = 0.999
        cost = v1v1
        return gamma, cost
    if v1v2 >= v2v2:
        # Case: Fig 1, first column
        gamma = 0.001
        cost = v2v2
        return gamma, cost
    # Case: Fig 1, second column
    gamma = (v2v2 - v1v2) / (v1v1 + v2v2 - 2 * v1v2)
    # v2v2 - gamm * gamma * (v1 - v2)^2
    # cost = v2v2 - gamma * gamma * (v1v1 + v2v2 - 2 * v1v2)
    #      = v2v2 - gamma * (v2v2 - v1v2)
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


phi1_last = 0
beta = 0.01


def fuse_gradients_single(g1, g2, fuse_manner):
    if fuse_manner == 'none':
        return g2
    if fuse_manner == 'average':
        return 0.5 * g1 + 0.5 * g2
    if fuse_manner == 'pareto':
        gamma, cost = _min_norm_element_from2(torch.dot(g1.float(), g1.float()),
                                              torch.dot(g1.float(), g2.float()),
                                              torch.dot(g2.float(), g2.float()))
        pareto_grad = gamma * g1 + (1 - gamma) * g2
        return pareto_grad
    if fuse_manner == 'pcgrad':
        g1p = g1 - torch.dot(g1.float(), g2.float()) / torch.dot(g2.float(), g2.float()) * g2
        return g1p + g2
    if fuse_manner == 'vaccine':
        phi = func.cosine_similarity(g1.float(), g2.float(), dim=0)
        global phi1_last
        phi1 = (1 - beta) * phi1_last + beta * phi
        phi1_last = phi1
        g1p = g1 + torch.norm(g1.float()) * (phi1 * torch.sqrt(1 - phi * phi) - phi * (torch.sqrt(1 - phi1 * phi1))) / (torch.norm(g2.float()) * torch.sqrt(1 - phi1 * phi1)) * g2
        return g1p + g2

    raise NotImplementedError(fuse_manner)


def fuse_gradients(g1, g2, fuse_manner):
    if g1 is None:
        return g2

    if isinstance(g1, dict):
        fused = {}
        for key, value in g1.items():
            fused[key] = fuse_gradients_single(g1[key], g2[key], fuse_manner)
        return fused
    else:
        return fuse_gradients_single(g1, g2, fuse_manner)


def catch_gradients(model):
    gradients = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gradients.append(p.grad)
    return parameters_to_vector(gradients) if gradients else None


def assign_gradients(model, gradients):
    offset = 0
    for name, p in model.named_parameters():
        numel = p.numel()
        p.grad = gradients[offset:offset + numel].view_as(p.data).clone()
        offset += numel
    assert offset == gradients.numel()


@register_task('batch_pareto_task', dataclass=BatchParetoConfig)
class BatchParetoTask(TranslationTask):

    def __init__(self, cfg: BatchParetoConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.fuse_manner = cfg.fuse_manner
        self.last_gradient = None

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)

        self.last_gradient = catch_gradients(model)
        model.zero_grad()
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)

        if self.last_gradient is not None:
            gradients = catch_gradients(model)
            fused_gradients = fuse_gradients(self.last_gradient, gradients, self.fuse_manner)
            assign_gradients(model, fused_gradients)

        # gradients = catch_gradients(model)
        # fused_gradients = fuse_gradients(self.last_gradient, gradients, self.fuse_manner)
        # assign_gradients(model, fused_gradients)

        # self.last_gradient = gradients

        return loss, sample_size, logging_output
