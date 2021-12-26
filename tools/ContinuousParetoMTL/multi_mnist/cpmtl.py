import random
from pathlib import Path
from termcolor import colored

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import transforms

import torch.utils.data

from tools.ContinuousParetoMTL.pareto.metrics import topk_accuracy
from tools.ContinuousParetoMTL.pareto.optim import MINRESKKTSolver
from tools.ContinuousParetoMTL.pareto.datasets import MultiMNIST
from tools.ContinuousParetoMTL.pareto.networks import MultiLeNet
from tools.ContinuousParetoMTL.pareto.utils import TopTrace


@torch.no_grad()
def evaluate(network, dataloader, device, closures, header=''):
    num_samples = 0
    losses = np.zeros(2)
    top1s = np.zeros(2)
    network.train(False)
    for images, labels in dataloader:
        batch_size = len(images)
        num_samples += batch_size
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        losses_batch = [c(network, logits, labels).item() for c in closures]
        losses += batch_size * np.array(losses_batch)
        top1s[0] += batch_size * topk_accuracy(logits[0], labels[:, 0], k=1)
        top1s[1] += batch_size * topk_accuracy(logits[1], labels[:, 1], k=1)
    losses /= num_samples
    top1s /= num_samples

    loss_msg = '[{}]'.format('/'.join([f'{loss:.6f}' for loss in losses]))
    top1_msg = '[{}]'.format('/'.join([f'{top1 * 100.0:.2f}%' for top1 in top1s]))
    msgs = [
        f'{header}:' if header else '',
        'loss', colored(loss_msg, 'yellow'),
        'top@1', colored(top1_msg, 'yellow')
    ]
    print(' '.join(msgs))
    return losses, top1s


def train(start_path, beta):

    # prepare hyper-parameters
    seed = 42
    cuda_enabled = True
    cuda_deterministic = False
    batch_size = 2048
    num_workers = 0
    shift = 0.0
    tol = 1e-5
    damping = 0.1
    max_iter = 50
    lr = 0.1
    momentum = 0.0
    weight_decay = 0.0
    num_steps = 10

    # prepare path
    ckpt_name = start_path.name.split('.')[0]
    root_path = Path(__file__).resolve().parent
    dataset_path = root_path / 'MultiMNIST'
    ckpt_path = root_path / 'cpmtl' / ckpt_name

    if not start_path.is_file():
        raise RuntimeError('Pareto solutions not found.')

    root_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    # fix random seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_enabled and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # prepare device

    if cuda_enabled and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        if cuda_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
    else:
        device = torch.device('cpu')

    # prepare dataset

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = MultiMNIST(dataset_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_set = MultiMNIST(dataset_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # prepare network
    network = MultiLeNet()
    network.to(device)

    # initialize network
    start_checkpoint = torch.load(start_path, map_location='cpu')
    network.load_state_dict(start_checkpoint['state_dict'])

    # prepare losses
    criterion = F.cross_entropy
    closures = [lambda n, l, t: criterion(l[0], t[:, 0]), lambda n, l, t: criterion(l[1], t[:, 1])]

    # prepare KKT solver
    kkt_solver = MINRESKKTSolver(
        network, device, train_loader, closures,
        shift=shift, tol=tol, damping=damping, maxiter=max_iter)

    # prepare optimizer
    optimizer = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # first evaluation
    losses, tops = evaluate(network, test_loader, device, closures, f'{ckpt_name}')

    # prepare utilities
    top_trace = TopTrace(len(closures))
    top_trace.print(tops, show=False)

    beta = beta.to(device)

    # training
    for step in range(1, num_steps + 1):
        network.train(True)
        optimizer.zero_grad()

        # 普通的优化：
        # loss = network(data)
        # loss.backward()

        kkt_solver.backward(beta)
        optimizer.step()
        losses, tops = evaluate(network, test_loader, device, closures, f'{ckpt_name}: {step}/{num_steps}')
        top_trace.print(tops)


def cpmtl():
    root_path = Path(__file__).resolve().parent
    start_root = root_path / 'weighted_sum'

    beta = torch.Tensor([1, 0])

    for start_path in sorted(start_root.glob('[0-9]*.pth'), key=lambda x: int(x.name.split('.')[0])):
        train(start_path, beta)


if __name__ == "__main__":
    cpmtl()
