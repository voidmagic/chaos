import collections
import logging

import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("sam_multi")
class SamTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    closures = []
    samples = []
    criterion = None
    grad_state = collections.defaultdict(dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--sam-adaptive', default='False', type=str)
        parser.add_argument('--sam-rho', default=0.05, type=float)
        TranslationMultiSimpleEpochTask.add_args(parser)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        adaptive = self.args.sam_adaptive == "True"
        sam_rho = self.args.sam_rho

        def step():
            model.train()
            model.set_num_updates(update_num)
            loss, sample_size, logging_output = criterion(model, sample)
            optimizer.backward(loss)
            return loss, sample_size, logging_output

        # first step, get gradient
        step()

        with torch.no_grad():
            grad_norm = optimizer.clip_grad_norm(max_norm=0.0)
            scale = sam_rho / (grad_norm + 1e-12)
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                self.grad_state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
            optimizer.zero_grad()

        loss, sample_size, logging_output = step()

        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                p.data = self.grad_state[p]["old_p"]  # get back to "w" from "w + e(w)"

        return loss, sample_size, logging_output
