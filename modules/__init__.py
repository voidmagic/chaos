from . import gptmt
from . import mtpt
from . import basics
from . import sync_mnmt
from . import auto_share
from . import temporal_attention
from . import transfer
from . import mtl_optim
from . import param_diff


import logging

import torch.multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)
