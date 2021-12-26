import torch
import torch.nn as nn
from torch import Tensor

from . import VisionHVPSolver
from .min_norm_solver import find_min_norm_element
from .linalg_solver import MINRESSolver


__all__ = ['MINRESKKTSolver']


class MINRESKKTSolver(object):
    def __init__(
            self,
            network: nn.Module,
            device: torch.device,
            train_loader,
            closures,
            *,
            shift: float = 0.0,
            tol: float = 1e-5,
            damping: float = 0.0,
            maxiter: int = 50) -> None:

        # prepare HVP solver
        hvp_solver = VisionHVPSolver(network, device, train_loader, closures)
        krylov_solver = MINRESSolver(network, hvp_solver, device, shift, tol, damping, maxiter)

        self.network = network
        self.hvp_solver = hvp_solver
        self.device = device
        self.krylov_solver = krylov_solver

    def backward(self, weights: Tensor) -> None:

        # jacobians alphas rhs
        jacobians = self.hvp_solver.grad_full()
        alphas, _ = find_min_norm_element(jacobians.detach())
        alphas = jacobians.new_tensor(alphas).detach()
        rhs = weights.view(1, -1).matmul(jacobians).view(-1).clone().detach()

        # explore
        lazy_jacobians = self.hvp_solver.grad_batch(create_graph=True)[0]
        with self.krylov_solver.solve(lazy_jacobians, jacobians, alphas, rhs) as results:
            direction, _ = results

        # apply gradient
        self.apply_grad(direction, normalize=True)

    @torch.no_grad()
    def apply_grad(self, direction: Tensor, *, normalize: bool = True) -> None:
        if normalize:
            direction.div_(direction.norm())
        offset = 0
        for p in self.hvp_solver.parameters:
            numel = p.numel()
            p.grad = direction[offset:offset + numel].view_as(p.data).clone()
            offset += numel
        assert offset == direction.size(0)

