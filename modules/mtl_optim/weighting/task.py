from fairseq.tasks import register_task

from modules.basics.sample_mnmt_single_model.task import SampledMultilingualSingleModelTask


@register_task('weighting_task')
class WeightMultilingualTask(SampledMultilingualSingleModelTask):
    weight = dict()

    @staticmethod
    def add_args(parser):
        SampledMultilingualSingleModelTask.add_args(parser)
        parser.add_argument('--weight', type=str, default=None)

    @classmethod
    def setup_task(cls, args, **kwargs):
        lang_pairs = args.lang_pairs.split(',')
        weight = ','.join(['1' for _ in lang_pairs]) if args.weight is None else args.weight
        weight = [float(w) for w in weight.split(',')]
        assert len(weight) == len(lang_pairs)
        weight = dict(zip(lang_pairs, weight))
        task:WeightMultilingualTask = super(WeightMultilingualTask, cls).setup_task(args)
        task.weight = weight
        return task

    def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
        loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
        loss *= self.weight[lang_pair]
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output
