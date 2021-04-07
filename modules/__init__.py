from . import google_mnmt
from . import inter_nat
from . import shallow_decoder
from . import gptmt
from . import mnmt_freeze
from . import bidir_trans
from . import mtpt
from . import translation
from . import positional_embedding
from . import sync_mnmt
from . import gptmt2
from . import auto_share
from . import gptmt3
from . import mnmt


# add preset arguments
import sys
if '--no-default' in sys.argv:
    sys.argv.remove('--no-default')
    pass
elif 'fairseq-train' in sys.argv[0]:
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
    if '--label-smoothing' not in sys.argv:
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

    # no left pad source
    if '--left-pad-source' not in sys.argv:
        sys.argv += ['--left-pad-source', 'False']

    # for debug
    if '--max-tokens' not in sys.argv and '--batch-size' not in sys.argv:
        sys.argv += ['--batch-size', '2']
