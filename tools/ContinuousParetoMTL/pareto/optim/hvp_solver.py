from typing import Tuple, List, Callable

import torch
import torch.nn as nn
import torch.utils.data
from torch import Tensor
from torch.nn.utils import parameters_to_vector


__all__ = ['VisionHVPSolver']


class VisionHVPSolver(object):
    """
    Hessian-Vector product calculation, Use PyTorch autograd for Hessian-Vector product calculation
    network:     PyTorch network to compute hessian for
    parameters:  parameters which are computed hessian w.r.t.
    dataloader:  PyTorch dataloader that we get examples from to compute grads
    device:      gpu/cpu device
    """

    def __init__(
            self,
            network: nn.Module,
            device: torch.device,
            dataloader: torch.utils.data.DataLoader,
            closures: List[Callable]) -> None:

        self.parameters = list(network.parameters())
        self.size = int(sum(p.numel() for p in self.parameters))
        self.network = network
        self.device = device
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataiter = iter(dataloader) if dataloader else None
        self.apply = self.apply_batch  # apply_full
        self.closures = closures

    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data.zero_()

    def set_data(self, dataloader: torch.utils.data.DataLoader) -> None:
        self.dataloader = dataloader
        self.dataiter = iter(dataloader)

    @torch.enable_grad()
    def grad_batch(self, *, create_graph: bool = False) -> Tuple[Tensor, List[Tensor]]:
        parameters = self.parameters
        losses = self.get_losses()
        param_grads = [list(torch.autograd.grad(
            loss, parameters,
            allow_unused=True, retain_graph=True, create_graph=create_graph)) for loss in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, parameters)):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
        grads = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)
        return grads, losses

    @torch.enable_grad()
    def grad_full(self, *, create_graph: bool = False) -> Tensor:
        num_batches = len(self.dataloader)
        grads = None
        for _ in range(num_batches):
            grads_batch, _ = self.grad_batch(create_graph=create_graph)
            if grads is None:
                grads = grads_batch
            else:
                grads.add_(grads_batch)
        grads.div_(num_batches)
        grads = grads.clone().detach()
        return grads

    @torch.enable_grad()
    def apply_batch(self, vec: Tensor, weights: Tensor = None, *, grads: Tensor = None, retain_graph: bool = True) -> Tuple[Tensor, Tensor]:

        """
        Returns H * vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """

        if grads is None:
            # compute original gradient, tracking computation graph
            self.zero_grad()
            grads, _ = self.grad_batch(create_graph=True)
            self.zero_grad()

        if weights is None:
            weighted_grad = grads.sum(dim=0)
        else:
            weighted_grad = torch.matmul(weights, grads)

        dot = vec.dot(weighted_grad)
        param_weighted_hvp = torch.autograd.grad(dot, self.parameters, retain_graph=retain_graph)

        # concatenate the results over the different components of the network
        weighted_hvp = parameters_to_vector([p.contiguous() for p in param_weighted_hvp])

        return weighted_hvp

    @torch.enable_grad()
    def apply_full(self, vec: Tensor, weights: Tensor = None, *, grads: Tensor = None, retain_graph: bool = False):
        num_batches = len(self.dataloader)
        weighted_hvp = None
        for _ in range(num_batches):
            weighted_hvp_batch = self.apply_batch(vec, weights, grads=grads, retain_graph=retain_graph)
            if weighted_hvp is None:
                weighted_hvp = weighted_hvp_batch
            else:
                weighted_hvp.add_(weighted_hvp_batch)
        weighted_hvp.div_(num_batches)
        return weighted_hvp

    @torch.enable_grad()
    def get_losses(self) -> List[Tensor]:

        try:
            inputs, targets = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            inputs, targets = next(self.dataiter)

        inputs = inputs.to(self.device)

        if isinstance(targets, list):
            targets = [target.to(self.device) for target in targets]
        else:
            targets = targets.to(self.device)

        logits = self.network(inputs)
        return [c(self.network, logits, targets) for c in self.closures]
