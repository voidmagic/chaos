import torch

from . import training
from . import clustering


torch.multiprocessing.set_sharing_strategy('file_system')
