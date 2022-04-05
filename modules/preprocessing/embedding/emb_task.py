import logging

import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("emb_task")
class EmbedTask(TranslationMultiSimpleEpochTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-mu', type=float, default=1.0)
        parser.add_argument('--emb-lambda', type=float, default=1.0)
        TranslationMultiSimpleEpochTask.add_args(parser)
        # fmt: on

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        try:
            loss += self.args.emb_mu * torch.norm(model.encoder.weight_s, p=1) + self.args.emb_lambda * torch.norm(model.encoder.weight_l, p=2)
        except OverflowError:
            pass
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
