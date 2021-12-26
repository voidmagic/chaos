
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import minres
from torch import Tensor

from .min_norm_solver import find_min_norm_element


__all__ = ['MINRESKKTSolver']


class MINRESKKTSolver(object):
    def __init__(self, network: nn.Module, train_loader, closures, *, shift: float = 0.0, tol: float = 1e-5, damping: float = 0.0, maxiter: int = 50):
        self.parameters = list(network.parameters())

        self.network = network
        self.closures = closures
        self.dataloader = train_loader
        self.dataiter = iter(train_loader) if train_loader else None

        self.linear_operator = HVPLinearOperator(network, damping)
        self.minres = partial(minres, shift=shift, tol=tol, maxiter=maxiter)

    def backward(self, weights: Tensor) -> None:

        # jacobians alphas rhs
        jacobians = self.grad_full()
        alphas, _ = find_min_norm_element(jacobians.detach())
        alphas = jacobians.new_tensor(alphas).detach()
        rhs = weights.view(1, -1).matmul(jacobians).view(-1).clone().detach()

        # explore
        lazy_jacobians = self.grad_batch(create_graph=True)
        direction = self.solve(lazy_jacobians, jacobians, alphas, rhs)

        # apply gradient
        self.apply_grad(direction, normalize=True)

    @torch.no_grad()
    def apply_grad(self, direction: Tensor, normalize: bool = True) -> None:
        if normalize:
            direction.div_(direction.norm())
        offset = 0
        for p in self.parameters:
            numel = p.numel()
            p.grad = direction[offset:offset + numel].view_as(p.data).clone()
            offset += numel
        assert offset == direction.size(0)

    @torch.enable_grad()
    def grad_batch(self, create_graph: bool = False):
        parameters = self.parameters
        losses = self.get_losses()
        param_grads = [list(torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True, create_graph=create_graph)) for loss in losses]
        for param_grad in param_grads:
            for i, (param_grad_module, param) in enumerate(zip(param_grad, parameters)):
                if param_grad_module is None:
                    param_grad[i] = torch.zeros_like(param)
        grads = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)
        return grads

    @torch.enable_grad()
    def grad_full(self, *, create_graph: bool = False) -> Tensor:
        num_batches = len(self.dataloader)
        grads = None
        for _ in range(num_batches):
            grads_batch = self.grad_batch(create_graph=create_graph)
            if grads is None:
                grads = grads_batch
            else:
                grads.add_(grads_batch)
        grads.div_(num_batches)
        return grads

    def solve(self, lazy_jacobians: Tensor, jacobians: Tensor, alphas: Tensor, rhs: Tensor):
        """Control counters automatically.

        Parameters
        ----------
        lazy_jacobians : torch.Tensor or None
            If not None, it is for gradient reusing. A matrix with shape (M,N).
        jacobians : torch.Tensor
            A matrix with shape (M,N). It should be identical to `rhs` and
            `lazy_jacobians` in this case (if `lazy_jacobians` is not None).
        alphas: torch.Tensor
            An array with shape (M,).
        rhs: torch.Tensor
            A matrix with shape (N,).
        """

        self.linear_operator.set_parameters(lazy_jacobians, alphas)
        x0 = jacobians.mean(0).neg().clone().detach().cpu().numpy()
        rhs = rhs.cpu().numpy()
        results = self.minres(self.linear_operator, rhs, x0=x0)
        d = torch.as_tensor(results[0]).cuda()
        return d

    @torch.enable_grad()
    def get_losses(self):

        try:
            inputs, targets = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.dataloader)
            inputs, targets = next(self.dataiter)

        inputs = inputs.cuda()

        if isinstance(targets, list):
            targets = [target.cuda() for target in targets]
        else:
            targets = targets.cuda()

        logits = self.network(inputs)
        return [c(self.network, logits, targets) for c in self.closures]


class HVPLinearOperator(LinearOperator):

    def __init__(self, network: nn.Module, damping: float) -> None:
        self.parameters = list(network.parameters())
        self.damping = damping
        size = int(sum(p.numel() for p in self.parameters))
        shape = (size, size)
        dtype = list(network.parameters())[0].detach().cpu().numpy().dtype

        super(HVPLinearOperator, self).__init__(dtype, shape)

        self.jacobians = None
        self.alphas = None

    def set_parameters(self, jacobians: Tensor, alphas: Tensor) -> None:
        self.jacobians = jacobians
        self.alphas = alphas

    def _matvec_tensor(self, tensor: Tensor):
        alphas_hvps = apply_batch(self.parameters, tensor, self.alphas, grads=self.jacobians, retain_graph=self.jacobians is not None)  # (N,)
        if self.damping > 0.0:
            alphas_hvps.add_(tensor, alpha=self.damping)
        return alphas_hvps

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        """HVP matrix-vector multiplication handler.

        If self is a linear operator of shape (N, N), then this method will
        be called on a shape (N,) or (N, 1) ndarray, and should return a
        shape (N,) or (N, 1) ndarray.

        In our case, it computes alpha_hession @ x.
        """
        tensor = torch.as_tensor(x.astype(self.dtype)).cuda()
        ret = self._matvec_tensor(tensor)
        return ret.detach().cpu().numpy()


@torch.enable_grad()
def apply_batch(parameters, vec: Tensor, weights: Tensor = None, *, grads: Tensor = None, retain_graph: bool = True):
    """
    Returns H * vec where H is the hessian of the loss w.r.t.
    the vectorized model parameters
    """

    if weights is None:
        weighted_grad = grads.sum(dim=0)
    else:
        weighted_grad = torch.matmul(weights, grads)

    dot = vec.dot(weighted_grad)
    param_weighted_hvp = torch.autograd.grad(dot, parameters, retain_graph=retain_graph)

    # concatenate the results over the different components of the network
    weighted_hvp = parameters_to_vector([p.contiguous() for p in param_weighted_hvp])

    return weighted_hvp
