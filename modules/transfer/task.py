from modules.basics.sample_mnmt.task import SampledMultilingualTask
from fairseq.tasks import register_task


@register_task('lrl_mt')
class TransferMultilingualTask(SampledMultilingualTask):
    @staticmethod
    def add_args(parser):
        SampledMultilingualTask.add_args(parser)
        parser.add_argument('--transfer-momentum', default=0.1, type=float, help='其他任务的动量')
        parser.add_argument('--transfer-target', type=str, help='目标任务')

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.transfer_target = args.transfer_target
        self.transfer_momentum = args.transfer_momentum
        self.last_sample = None

    def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
        loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
        if ignore_grad:
            loss *= 0
        if lang_pair != self.transfer_target:
            loss *= self.transfer_momentum
        optimizer.backward(loss)
        return loss, sample_size, logging_output
