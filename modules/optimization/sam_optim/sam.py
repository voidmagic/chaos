import collections

from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.optim
from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig


@dataclass
class SamAdamConfig(FairseqAdamConfig):
    sam_rho: float = field(default=0.05, metadata={"help": "Sam RHO"})
    sam_adaptive: bool = field(default=False, metadata={"help": "Use adaptive sam"})


@register_optimizer("samadam", dataclass=SamAdamConfig)
class SamAdamOptimizer(FairseqAdam):
    state = collections.defaultdict(dict)

    def __init__(self, cfg: SamAdamConfig, params):
        super(SamAdamOptimizer, self).__init__(cfg, params)
        self.rho = cfg.sam_rho
        self.adaptive = cfg.sam_adaptive

    @torch.no_grad()
    def first_step(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

    @torch.no_grad()
    def step(self, closure=None, scale=1.0, groups=None):
        self.first_step()
        self.zero_grad()
        with torch.enable_grad():
            for c in closure or []:
                c()
        self.second_step()
        super(SamAdamOptimizer, self).step(scale=scale)

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
