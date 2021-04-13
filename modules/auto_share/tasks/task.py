import contextlib
import logging
import os

import torch
import random
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.trainer import Trainer

from .dataset import FastRoundRobinDataset
from .view import ModelView

logger = logging.getLogger(__name__)


@register_task('auto_share')
class AutoShareTranslationTask(MultilingualTranslationTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.view = None
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.start_split = int(os.environ.get('SPLIT_START', '10'))
        self.split_every = int(os.environ.get('SPLIT_EVERY', '1'))
        self.grad_valid = os.environ.get('GRAD_VALID', 'multi')

        self.sample_method = os.environ.get('SAMPLE_METHOD', 'uniform')
        self.sample_temperature = int(os.environ.get('TEMPERATURE', '5'))
        assert self.sample_method in ['uniform', 'temperature', 'proportional']
        if self.sample_method == 'proportional':
            self.sample_temperature = 1  # 等价

    def build_model(self, args):
        model = super(AutoShareTranslationTask, self).build_model(args)
        self.view = ModelView(model)

        # 加载训练好的模型
        model_path = os.environ.get('MULTI_MODEL', None)
        if model_path is not None:
            logger.info('load pretrain states from {}'.format(model_path))
            states = torch.load(model_path)['model']
            model.load_state_dict(states)
            self.start_split = 0
        return model

    def begin_epoch(self, epoch, model):
        if epoch < self.start_split or self.view is None:
            return
        if epoch % self.split_every != 1 and self.split_every != 1:
            # 1. 每split_every个epoch运行一次，否则返回（!=1因为epoch从1开始）
            # 2. 如果split_every为1，每次都运行
            return

        trainer = get_trainer()
        criterion = trainer.criterion
        optimizer = trainer.optimizer

        logger.info("Start parameter sharing")
        # requires: criterion optimizer
        model.train()

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
        for i, sample in enumerate(batch_iterator):
            if self.cuda:
                sample = utils.move_to_cuda(sample)
            for lang_pair in self.lang_pairs:
                loss, _, _ = criterion(model.models[lang_pair], sample[lang_pair])
                optimizer.backward(loss)
                self.view.accum_gradient(lang_pair)
                model.zero_grad()
        self.view.auto_split()
        trainer.reinitialize()

    def load_dataset(self, split, epoch=1, **kwargs):
        super(AutoShareTranslationTask, self).load_dataset(split, epoch)
        if split == 'train':
            old_dataset: RoundRobinZipDatasets = self.datasets[split]
            self.datasets[split] = FastRoundRobinDataset(datasets=old_dataset.datasets, eval_key=old_dataset.eval_key)

    def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
        ignore_grad = ignore_grad or self.sampler(lang_pair)
        return super(AutoShareTranslationTask, self)._per_lang_pair_train_loss(
            lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad)

    def sampler(self, lang_pair):
        if self.sample_method == 'uniform':
            return True
        sizes = {
            key: len(self.dataset('train').datasets[key]) ** (1 / self.sample_temperature)
            for key in self.dataset('train').datasets.keys()
        }
        p_scale = sizes[lang_pair] / max(sizes.values())
        return random.random() > p_scale


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
