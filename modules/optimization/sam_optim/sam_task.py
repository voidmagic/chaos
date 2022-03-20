import logging

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("sam_multi")
class SamTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        def closure():
            model.train()
            model.set_num_updates(update_num)
            loss, sample_size, logging_output = criterion(model, sample)
            optimizer.backward(loss)
            return loss

        optimizer.set_closure(closure)
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        optimizer.backward(loss)
        return loss, sample_size, logging_output
