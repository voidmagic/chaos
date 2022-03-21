import logging

from fairseq.optim import FP16Optimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("sam_multi")
class SamTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    closures = []

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.closures.append(lambda: optimizer.backward(criterion(model, sample)[0]))

        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def optimizer_step(self, optimizer, model, update_num):
        optimizer.step(closure=self.closures)
        self.closures = []
