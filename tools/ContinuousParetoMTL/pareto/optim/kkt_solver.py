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
            hvp_solver: VisionHVPSolver,
            device: torch.device,
            *,
            stochastic: bool = True,
            kkt_momentum: float = 0.0,
            create_graph: bool = False,
            grad_correction: bool = False,
            shift: float = 0.0,
            tol: float = 1e-5,
            damping: float = 0.0,
            maxiter: int = 50) -> None:

        krylov_solver = MINRESSolver(network, hvp_solver, device, shift, tol, damping, maxiter)
        self.network = network
        self.hvp_solver = hvp_solver
        self.device = device
        self.kkt_momentum = kkt_momentum
        self.jacobians_momentum_buffer = None
        self.alphas_momentum_buffer = None
        self.create_graph = create_graph
        self.grad_correction = grad_correction
        self.stochastic = stochastic
        self.krylov_solver = krylov_solver

    def zero_grad(self) -> None:
        self.hvp_solver.zero_grad()

    def backward(self, weights: Tensor) -> None:

        # jacobians alphas rhs
        jacobians = self.hvp_solver.grad(create_graph=self.create_graph)
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

