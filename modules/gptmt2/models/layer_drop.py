# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
LayerDrop as described in https://arxiv.org/abs/1909.11556.
"""

import torch
import torch.nn as nn
from threading import Lock


class LayerDropModuleList(nn.ModuleList):
    """
    A LayerDrop implementation based on :class:`torch.nn.ModuleList`.

    We refresh the choice of which layers to drop every time we iterate
    over the LayerDropModuleList instance. During evaluation we always
    iterate over all layers.

    Usage::

        layers = LayerDropList(p=0.5, modules=[layer1, layer2, layer3])
        for layer in layers:  # this might iterate over layers 1 and 3
            x = layer(x)
        for layer in layers:  # this might iterate over all layers
            x = layer(x)
        for layer in layers:  # this might not iterate over any layers
            x = layer(x)

    Args:
        p (float): probability of dropping out each layer
        modules (iterable, optional): an iterable of modules to add
    """

    def __init__(self, p, modules=None):
        super().__init__(modules)
        self.p = p
        self.prob_obj = Singleton()

    # def __iter__(self):
    #     dropout_probs = torch.empty(len(self)).uniform_()
    #     for i, m in enumerate(super().__iter__()):
    #         if not self.training or (dropout_probs[i] > self.p):
    #             yield m

    def with_drop(self, reset):
        if reset:
            self.prob_obj.reset(len(self))
        for i, m in enumerate(super().__iter__()):
            if not self.training or (self.prob_obj.get_value()[i] > self.p):
                yield m
            else:
                yield sum([torch.sum(p) for p in m.parameters()]) * 0.


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = None

    def get_value(self):
        return self.value

    def reset(self, n):
        self.value = torch.empty(n).uniform_()
