from contextlib import contextmanager
from functools import partial
from typing import Tuple

import numpy as np

from scipy.sparse.linalg import LinearOperator, minres

import torch
import torch.nn as nn
from torch import Tensor

from .hvp_solver import VisionHVPSolver


__all__ = ['HVPLinearOperator', 'MINRESSolver']


class HVPLinearOperator(LinearOperator):

    def __init__(self, network: nn.Module, hvp_solver: VisionHVPSolver, damping: float) -> None:
        shape = (hvp_solver.size, hvp_solver.size)
        dtype = list(network.parameters())[0].detach().cpu().numpy().dtype

        super(HVPLinearOperator, self).__init__(dtype, shape)

        self.network = network
        self.hvp_solver = hvp_solver
        self.damping = damping

        self.jacobians = None
        self.alphas = None
        self.reset_parameters()

        self.hvp_counter = 0
        self.matvec_counter = 0
        self.reset_counters()

    def set_parameters(self, jacobians: Tensor, alphas: Tensor) -> None:
        self.jacobians = jacobians
        self.alphas = alphas

    def reset_parameters(self) -> None:
        self.jacobians = None
        self.alphas = None

    def reset_counters(self) -> None:
        self.hvp_counter = 0
        self.matvec_counter = 0

    def get_counters(self) -> Tuple[int, int]:
        return self.hvp_counter, self.matvec_counter

    def _matvec_tensor(self, tensor: Tensor):
        alphas_hvps = self.hvp_solver.apply_batch(tensor, self.alphas, grads=self.jacobians, retain_graph=self.jacobians is not None)  # (N,)
        if self.damping > 0.0:
            alphas_hvps.add_(tensor, alpha=self.damping)
        self.hvp_counter += 1
        self.matvec_counter += 1
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


class MINRESSolver(object):

    def __init__(self, network: nn.Module, hvp_solver: VisionHVPSolver, shift: float, tol: float, damping: float, maxiter: int) -> None:
        self.linear_operator = HVPLinearOperator(network, hvp_solver, damping)
        self.minres = partial(minres, shift=shift, tol=tol, maxiter=maxiter)
        self.shape = self.linear_operator.shape
        self.dtype = self.linear_operator.dtype

    @contextmanager
    def solve(self, lazy_jacobians: Tensor, jacobians: Tensor, alphas: Tensor, rhs: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
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

        try:
            self.linear_operator.set_parameters(lazy_jacobians, alphas)
            x0 = jacobians.mean(0).neg().clone().detach().cpu().numpy()
            rhs = rhs.cpu().numpy()
            results = self.minres(self.linear_operator, rhs, x0=x0)
            d = torch.as_tensor(results[0].astype(self.dtype)).cuda()
            yield d, self.linear_operator.get_counters()
        finally:
            self.linear_operator.reset_parameters()
            self.linear_operator.reset_counters()
