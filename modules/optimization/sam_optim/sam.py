import collections

from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.optim
from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig
from omegaconf import II


@dataclass
class SamAdamConfig(FairseqAdamConfig):
    adam_betas: Any = field(
        default=(0.9, 0.999), metadata={"help": "betas for Adam optimizer"}
    )
    adam_eps: float = field(
        default=1e-8, metadata={"help": "epsilon for Adam optimizer"}
    )
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    fp16_adam_stats: bool = field(
        default=False, metadata={"help": "use FP16 stats (with automatic scaling)"}
    )
    # TODO common vars below in parent
    tpu: bool = II("common.tpu")
    lr: List[float] = II("optimization.lr")

    sam_rho: float = field(
        default=0.05, metadata={"help": "Sam RHO"}
    )
    sam_adaptive: bool = field(
        default=False, metadata={"help": "Use adaptive sam"}
    )


@register_optimizer("samadam", dataclass=SamAdamConfig)
class SamAdamOptimizer(FairseqAdam):
    closure = None
    state = collections.defaultdict(dict)

    def __init__(self, cfg: SamAdamConfig, params):
        super(SamAdamOptimizer, self).__init__(cfg, params)
        self.rho = cfg.sam_rho
        self.adaptive = cfg.sam_adaptive

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = self.rho / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # self.base_optimizer.step()  # do the actual "sharpness-aware" update
        super(SamAdamOptimizer, self).step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        closure = closure or self.closure
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

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

    def set_closure(self, closure):
        self.closure = closure


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
