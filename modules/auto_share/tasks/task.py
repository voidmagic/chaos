import logging
import os

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.trainer import Trainer
from modules.sample_mnmt.task import SampledMultilingualTask
from modules.auto_share.tasks.view import ModelView

logger = logging.getLogger(__name__)


@register_task('auto_share')
class AutoShareTranslationTask(SampledMultilingualTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.view = None
        self.cuda = torch.cuda.is_available() and not args.cpu

        # 以下是自定义参数
        self.split_counter = LoopCounter(int(os.environ.get('SPLIT_EVERY', '5')))
        self.grad_valid = os.environ.get('GRAD_VALID', 'multi')

        self.split_all = os.environ.get('SPLIT_ALL', 'FALSE') == 'TRUE'
        self.threshold = float(os.environ.get('THRESHOLD', '0.0'))

        # 可以是parameter，module，layer
        self.granularity = os.environ.get('GRANULARITY', 'parameter')

    def build_model(self, args):
        model = super(AutoShareTranslationTask, self).build_model(args)
        self.view = ModelView(model, split_all=self.split_all, threshold=self.threshold, granularity=self.granularity)
        return model

    def begin_valid_epoch(self, epoch, model):
        trainer = get_trainer()
        criterion = trainer.criterion
        optimizer = trainer.optimizer

        logger.info("Start accumulating gradient")
        if self.grad_valid in self.datasets:
            dataset_for_split = self.dataset(self.grad_valid)
        else:
            try:
                self.load_dataset(self.grad_valid)
                dataset_for_split = self.dataset(self.grad_valid)
            except FileNotFoundError:
                dataset_for_split = self.dataset('valid')

        batch_iterator = self.get_batch_iterator(
            dataset=dataset_for_split,
            max_tokens=self.args.max_tokens_valid,
            max_sentences=self.args.batch_size_valid,
            max_positions=utils.resolve_max_positions(self.max_positions(), model.max_positions()),
            ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=self.args.required_batch_size_multiple,
            seed=self.args.seed,
            num_workers=self.args.num_workers,
            data_buffer_size=self.args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)

        model.eval()
        for i, sample in enumerate(batch_iterator):
            if self.cuda:
                sample = utils.move_to_cuda(sample)
            for lang_pair in self.lang_pairs:
                loss, _, _ = criterion(model.models[lang_pair], sample[lang_pair])
                # 缩放一下，避免出现NAN
                loss = loss / len(batch_iterator) / self.split_counter
                optimizer.backward(loss)
                self.view.accum_gradient(lang_pair)
                model.zero_grad()
        model.train()

        self.split_counter += 1
        if self.split_counter == 0:
            self.view.auto_split()  # 切分参数
            trainer.reinitialize()  # 把所有参数加入优化器
            logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj


class LoopCounter:
    def __init__(self, n):
        self.n = n
        self.current = 0

    def __add__(self, other):
        self.current = (self.current + other) % self.n
        return self

    def __eq__(self, other):
        return self.current == other

    def __rtruediv__(self, other):
        return other / self.n
