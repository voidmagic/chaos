from fairseq.file_io import PathManager

from . import google_mnmt
from . import gptmt
from . import mtpt
from . import translation
from . import sync_mnmt
from . import auto_share
from . import sample_mnmt
from . import temporal_attention
from . import transfer
from . import pcgrad
import os
import logging
from fairseq import checkpoint_utils

import torch.multiprocessing
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.multiprocessing.set_sharing_strategy('file_system')

logger = logging.getLogger(__name__)


def save_with_remove(obj, f):
    if isinstance(f, str):
        os.remove(f)
        logger.info("Remove file {} that exists".format(f))
        with PathManager.open(f, "wb") as h:
            save_with_remove(obj, h)
    else:
        return torch.save(obj, f)


checkpoint_utils.torch_persistent_save = save_with_remove
