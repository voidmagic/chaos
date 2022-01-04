import logging
from fairseq import utils
from .view import ModelView
from fairseq.trainer import Trainer
import os
from collections import OrderedDict

from fairseq.data import RoundRobinZipDatasets, indexed_dataset, PrependTokenDataset, data_utils, LanguagePairDataset
from fairseq.tasks import register_task

from modules.basics.sample_mnmt.task import SampledMultilingualTask
from modules.basics.sample_mnmt.dataset import MultilingualSampledDataset

logger = logging.getLogger(__name__)


@register_task('parameter_differentiation_xd')
class ParameterDifferentiationXdTask(SampledMultilingualTask):
    _view: ModelView = None
    _initial_size = None
    _trained_step = 0

    @property
    def view(self):
        if self._view is None:
            self._view = ModelView(get_trainer().model)
        return self._view

    def record_gradient(self, model):
        logger.info("Start accumulating gradient")
        criterion = get_trainer().get_criterion()
        model.eval()  # disable dropout
        for lang_pair, dataset in self.dataset(self.args.valid_subset).datasets.items():
            batch_iterator = self.get_batch_iterator(
                dataset=dataset, max_tokens=self.args.max_tokens_valid, seed=self.args.seed).next_epoch_itr()
            model.zero_grad()
            for sample in batch_iterator:
                sample = utils.move_to_cuda(sample)
                loss, _, _ = criterion(model.models[lang_pair], sample)
                loss = loss / len(batch_iterator)
                loss.backward()
            self.view.accum_gradient(lang_pair)
            model.zero_grad()
        model.train()  # enable dropout
        logger.info("End accumulating gradient")

    def begin_valid_epoch(self, epoch, model):
        if self._initial_size is None:
            self._initial_size = sum(p.numel() for p in model.parameters())

        current_size = sum(p.numel() for p in model.parameters())
        if current_size / self._initial_size > 1.5:
            return

        self.record_gradient(model)
        logger.info("num. model params before: {}".format(sum(p.numel() for p in model.parameters())))
        _ = list(self.view.auto_split())
        logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))
        self.view.clear_gradient()
        get_trainer()._optimizer = None

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.begin_valid_epoch(epoch=0, model=model)

        current_size = sum(p.numel() for p in model.parameters())
        if self._initial_size and current_size / self._initial_size < 1.5:
            ignore_grad = True

        if self._trained_step > 5:
            ignore_grad = True

        if not ignore_grad:
            self._trained_step += 1
        return super(ParameterDifferentiationXdTask, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def load_dataset(self, split, epoch=1, **kwargs):
        # 与gmnmt兼容
        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.dicts[src])
            return (PrependTokenDataset(src_raw_dataset, self.dicts[src].index('__{}__'.format(tgt))),
                    data_utils.load_indexed_dataset(prefix + tgt, self.dicts[tgt]))

        def load_language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_prepend_dataset, tgt_raw_dataset = load_data(src, tgt)
            dataset = LanguagePairDataset(src_prepend_dataset, src_prepend_dataset.sizes, self.dicts[src], tgt_raw_dataset, tgt_raw_dataset.sizes, self.dicts[tgt])
            return dataset

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(
                [
                    (lang_pair, load_language_pair_dataset(lang_pair))
                    for lang_pair in self.lang_pairs
                ]
            ),
            eval_key=None
            if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )
        if split != 'train':
            return
        self.datasets[split] = MultilingualSampledDataset(
            self.datasets[split].datasets,
            self.datasets[split].eval_key,
            sample_method=self.args.sample_method,
            temperature=self.args.sample_temperature
        )


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj



