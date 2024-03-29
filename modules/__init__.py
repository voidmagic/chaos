# from . import gpt
import logging

from . import basics
from . import preprocessing
from . import training
from . import optimization
from . import generation

import torch.multiprocessing
import warnings


warnings.filterwarnings("ignore", category=UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

# disable some logging
logger = logging.getLogger('fairseq.tasks.translation_multi_simple_epoch')
logger.setLevel(logging.WARN)
