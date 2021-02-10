from . import google_mnmt
from . import label_smooth
from . import inter_nat
from . import shallow_decoder
from . import gptmt
from . import mnmt_fix
from . import bidir_trans
from . import mtpt
from . import translation


# add preset arguments
import sys
if 'fairseq-train' in sys.argv:
    # --optimizer adam
    if '--optimizer' not in sys.argv:
        sys.argv += ['--optimizer', 'adam']

    # --adam-betas '(0.9, 0.98)'
    if '--adam-betas' not in sys.argv:
        sys.argv += ['--adam-betas', '(0.9, 0.98)']

    # --clip-norm 0.0
    if '--clip-norm' not in sys.argv:
        sys.argv += ['--clip-norm', '0.0']

    # --lr 0.0007
    if '--lr' not in sys.argv:
        sys.argv += ['--lr', '0.0007']

    # --min-lr 1e-09
    if '--min-lr' not in sys.argv:
        sys.argv += ['--min-lr', '1e-09']

    # --weight-decay 0.0001
    if '--weight-decay' not in sys.argv:
        sys.argv += ['--weight-decay', '0.0001']

    # --fp16-scale-tolerance 0.25
    if '--fp16-scale-tolerance' not in sys.argv:
        sys.argv += ['--fp16-scale-tolerance', '0.25']

    # --lr-scheduler inverse_sqrt
    if '--lr-scheduler' not in sys.argv:
        sys.argv += ['--lr-scheduler', 'inverse_sqrt']

    # --warmup-updates 4000
    if '--warmup-updates' not in sys.argv:
        sys.argv += ['--warmup-updates', '4000']

    # --warmup-init-lr 1e-07
    if '--warmup-init-lr' not in sys.argv:
        sys.argv += ['--warmup-init-lr', '1e-07']

    # --criterion label_smoothed_cross_entropy
    if '--criterion' not in sys.argv:
        sys.argv += ['--criterion', 'label_smoothed_cross_entropy']

    # --label-smoothing 0.1
    if '--criterion' not in sys.argv:
        sys.argv += ['--label-smoothing', '0.1']

    # --no-progress-bar
    if '--no-progress-bar' not in sys.argv:
        sys.argv += ['--no-progress-bar']

    # --no-epoch-checkpoints
    if '--no-epoch-checkpoints' not in sys.argv:
        sys.argv += ['--no-epoch-checkpoints']

    # save dir for debug
    if '--save-dir' not in sys.argv:
        sys.argv += ['--save-dir', 'data/ckpts']

    # resize batch size
    import torch
    device_count = max(1, torch.cuda.device_count())
    if '--max-tokens' in sys.argv:
        max_tokens = int(sys.argv[sys.argv.index('--max-tokens') + 1])
        max_tokens = int(max_tokens / device_count)
        sys.argv[sys.argv.index('--max-tokens') + 1] = str(max_tokens)
    elif '--batch-size' in sys.argv:
        batch_size = int(sys.argv[sys.argv.index('--batch-size') + 1])
        batch_size = int(batch_size / device_count)
        sys.argv[sys.argv.index('--batch-size') + 1] = str(batch_size)
    else:
        # for debug
        sys.argv += ['--batch-size', '2']
