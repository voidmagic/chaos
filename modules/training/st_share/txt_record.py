import collections
import logging
import os

import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("record_txt_task")
class RecordTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    gradients = collections.defaultdict(list)

    def valid_step(self, sample, model, criterion):

        model.eval()
        model.zero_grad()
        loss, sample_size, logging_output = criterion(model, sample)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.gradients[name].append(p.grad)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super(TranslationMultiSimpleEpochTask, self).reduce_metrics(logging_outputs, criterion)
        filename = os.environ.get("gradient_path", default="{}.pt".format(list(self.datasets.keys())[0]))
        torch.save({key: torch.mean(torch.stack(value, dim=0), dim=0).view(-1).cpu() for key, value in self.gradients.items()}, filename)
