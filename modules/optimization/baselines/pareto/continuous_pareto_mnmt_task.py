import logging

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from modules.optimization.baselines.pareto.optim import VisionHVPSolver, MINRESKKTSolver

logger = logging.getLogger(__name__)


@register_task("cpareto_mnmt")
class ContinuousParetoMultilingualNeuralMachineTranslationTask(TranslationMultiSimpleEpochTask):
    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)

        loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0

        optimizer.backward(loss)
        return loss, sample_size, logging_output
