import numpy as np

import torch
import torch.utils.data

from model_lenet import RegressionModel, RegressionTrain

from min_norm_solvers import get_d_paretomtl_init, get_d_paretomtl

from utils import load_dataset, circle_points


def find_init_solution(model, optimizer, train_loader, n_tasks, ref_vec, pref_idx):
    for t in range(2):
        model.train()
        for (it, (x, ts)) in enumerate(train_loader):
            x = x.cuda()
            ts = ts.cuda()
            grads = dict()
            losses_vec = []

            # obtain and store the gradient value
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(x, ts)
                losses_vec.append(task_loss[i].data)
                task_loss[i].backward()

                grads[i] = [p.grad.clone().detach().flatten() for p in model.parameters() if p.grad is not None]

            grads = torch.stack([torch.cat(grads[i]) for i in range(len(grads))])

            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            flag, weight_vec = get_d_paretomtl_init(grads, losses_vec, ref_vec, pref_idx)

            # early stop once a feasible solution is obtained
            if flag:
                print("feasible solution is obtained.")
                break

            # optimization step
            loss_total = 0
            optimizer.zero_grad()
            for i in range(n_tasks):
                task_loss = model(x, ts)
                loss_total = loss_total + weight_vec[i] * task_loss[i]
            loss_total.backward()
            optimizer.step()


def run_pareto():
    pass


def train(niter, init_weight, pref_idx, n_tasks=2):
    # generate #npref preference vectors
    ref_vec = circle_points(5)
    train_loader, test_loader = load_dataset()

    model = RegressionTrain(RegressionModel(n_tasks), init_weight).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15,30,45,60,75,90], gamma=0.5)

    # store information during optimization
    weights, task_train_losses, train_accs = [], [], []

    find_init_solution(model, optimizer, train_loader, n_tasks, ref_vec, pref_idx)

    # run niter epochs of ParetoMTL 
    for t in range(niter):
        scheduler.step()
        model.train()
        for (it, (x, ts)) in enumerate(train_loader):
            x = x.cuda()
            ts = ts.cuda()
            grads = dict()
            losses_vec = []
            
            for i in range(n_tasks):
                optimizer.zero_grad()
                task_loss = model(x, ts)
                losses_vec.append(task_loss[i].data)
                task_loss[i].backward()
                grads[i] = [p.grad.clone().detach().flatten() for p in model.parameters() if p.grad is not None]

            grads_list = [torch.cat(grads[i]) for i in range(len(grads))]
            grads = torch.stack(grads_list)
            
            # calculate the weights
            losses_vec = torch.stack(losses_vec)
            weight_vec = get_d_paretomtl(grads, losses_vec, ref_vec, pref_idx)
            
            normalize_coeff = n_tasks / torch.sum(torch.abs(weight_vec))
            weight_vec = weight_vec * normalize_coeff
            
            # optimization step
            optimizer.zero_grad()
            loss_total = 0
            for i in range(n_tasks):
                task_loss = model(x, ts)
                loss_total = loss_total + weight_vec[i] * task_loss[i]
            loss_total.backward()
            optimizer.step()


train(niter=100, init_weight=np.array([0.5 , 0.5 ]), pref_idx=2)