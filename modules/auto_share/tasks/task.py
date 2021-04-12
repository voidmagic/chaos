import os
import logging
from collections import OrderedDict

import torch
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.trainer import Trainer

from .dataset import TemperatureRoundRobinDataset
from .view import ModelView


logger = logging.getLogger(__name__)


@register_task('auto_share')
class AutoShareTranslationTask(MultilingualTranslationTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.view = None
        self.start_split = int(os.environ.get('SPLIT_START', '10'))
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.split_every = int(os.environ.get('SPLIT_EVERY', '1'))
        self.grad_valid = os.environ.get('GRAD_VALID', 'multi')

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
        """Load a dataset split."""
        # 开始
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split("-")
            langpair_dataset = load_langpair_dataset(
                data_path,
                split,
                src,
                self.dicts[src],
                tgt,
                self.dicts[tgt],
                combine=True,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(
                [
                    (lang_pair, language_pair_dataset(lang_pair))
                    for lang_pair in self.lang_pairs
                ]
            ),
            eval_key=None
            if self.training
            else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )
        # 结束

        # 使用重写的TemperatureRoundRobinDataset
        if split == 'train':
            self.datasets[split] = TemperatureRoundRobinDataset(
                OrderedDict(
                    [
                        (lang_pair, language_pair_dataset(lang_pair))
                        for lang_pair in self.lang_pairs
                    ]
                ),
                eval_key=None
                if self.training
                else "%s-%s" % (self.args.source_lang, self.args.target_lang),
            )



def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
