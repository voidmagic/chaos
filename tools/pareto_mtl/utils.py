import pickle
import numpy as np

import torch
import torch.utils.data


def circle_points(n):
    t = np.linspace(0, 0.5 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    ref_vec = torch.tensor(np.c_[x, y]).cuda().float()
    return ref_vec


def load_dataset():
    # load dataset
    with open('data/multi_mnist.pickle', 'rb') as f:
        train_x, train_label, test_x, test_label = pickle.load(f)

    train_x = torch.from_numpy(train_x.reshape(120000, 1, 36, 36)).float()
    train_label = torch.from_numpy(train_label).long()
    test_x = torch.from_numpy(test_x.reshape(20000, 1, 36, 36)).float()
    test_label = torch.from_numpy(test_label).long()

    train_set = torch.utils.data.TensorDataset(train_x, train_label)
    test_set = torch.utils.data.TensorDataset(test_x, test_label)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)

    print('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}'.format(len(test_loader)))
    return train_loader, test_loader