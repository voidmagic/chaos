import torch
from functools import partial
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector
from scipy.sparse.linalg import minres, LinearOperator
from modules.optimization.baselines.pareto.min_norm_solver import find_min_norm_element


def mat_vec(x, weighted_grad, parameters):
    x = torch.tensor(x).float().cuda()
    dot = x.dot(weighted_grad)
    param_weighted_hvp = torch.autograd.grad(dot, parameters, retain_graph=True)
    alphas_hvps = parameters_to_vector([p.contiguous() for p in param_weighted_hvp])
    alphas_hvps.add_(x, alpha=0.1)
    return alphas_hvps.detach().cpu().numpy()


def calculate_jacobian(data, network, create_graph=False):
    inputs, targets = data
    inputs = inputs.cuda()
    targets = targets.cuda()
    logits = network(inputs)
    losses = [F.cross_entropy(logits[0], targets[:, 0]), F.cross_entropy(logits[1], targets[:, 1])]

    parameters = list(network.parameters())

    grad_func = partial(torch.autograd.grad, allow_unused=True, retain_graph=True, create_graph=create_graph)
    param_grads = [list(grad_func(loss, parameters)) for loss in losses]
    for i, original_grads in enumerate(param_grads):
        param_grads[i] = [torch.zeros_like(p) if g is None else g for g, p in zip(original_grads, parameters)]
    grads_jacobian = torch.stack([parameters_to_vector(param_grad) for param_grad in param_grads], dim=0)
    return grads_jacobian


def pareto_backward(network, weights, train_loader):
    n_jacobian = 30
    n_hessian = 2

    parameters = list(network.parameters())

    # jacobians alphas rhs
    dataiter = iter(train_loader)
    jacobians = [calculate_jacobian(next(dataiter), network) for _ in range(n_jacobian)]
    jacobians = torch.mean(torch.stack(jacobians), dim=0)
    alphas, _ = find_min_norm_element(jacobians.detach())
    alphas = torch.tensor(alphas).float().to(jacobians.device)
    rhs = weights.view(1, -1).matmul(jacobians).view(-1)

    # explore
    dataiter = iter(train_loader)

    jacobians = [calculate_jacobian(next(dataiter), network, create_graph=True) for _ in range(n_hessian)]
    jacobians = torch.mean(torch.stack(jacobians), dim=0)
    mat_vec_partial = partial(mat_vec, weighted_grad=torch.matmul(alphas, jacobians), parameters=parameters)

    linear_operator = LinearOperator((jacobians.size(1), jacobians.size(1)), matvec=mat_vec_partial)
    results, _ = minres(linear_operator, rhs.cpu().numpy(), maxiter=50)
    direction = torch.tensor(results).cuda()

    # apply gradient
    direction.div_(direction.norm())
    offset = 0
    for p in parameters:
        numel = p.numel()
        p.grad = direction[offset:offset + numel].view_as(p.data).clone()
        offset += numel
