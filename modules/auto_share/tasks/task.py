import logging
import os

import torch
from fairseq import utils
from fairseq.data import RoundRobinZipDatasets, iterators, data_utils
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.trainer import Trainer

from .dataset import FastRoundRobinDataset
from .sample_iterator import MyEpochBatchIterator
from .view import ModelView

logger = logging.getLogger(__name__)


@register_task('auto_share')
class AutoShareTranslationTask(MultilingualTranslationTask):
    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        if hasattr(args, 'distributed_num_procs'):  # training
            assert args.distributed_num_procs == 1, "目前只能单卡训练，多卡在某些情况会卡死，且多卡情况下参数拆分后不同卡上的新参数无法同步"
        self.src_dict = self.tgt_dict = list(dicts.values())[0]
        self.view = None
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.split_counter = LoopCounter(int(os.environ.get('SPLIT_EVERY', '5')))
        self.grad_valid = os.environ.get('GRAD_VALID', 'multi')

        self.sample_method = os.environ.get('SAMPLE_METHOD', 'uniform')
        self.sample_temperature = int(os.environ.get('TEMPERATURE', '5'))
        assert self.sample_method in ['uniform', 'temperature', 'proportional']
        if self.sample_method == 'proportional':
            self.sample_method = 'temperature'
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
        return model

    def begin_valid_epoch(self, epoch, model):
        trainer = get_trainer()
        criterion = trainer.criterion
        optimizer = trainer.optimizer

        logger.info("Start accumulating gradient")
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
                # 缩放一下，避免出现NAN
                loss = loss / len(batch_iterator) / self.split_counter
                optimizer.backward(loss)
                self.view.accum_gradient(lang_pair)
                model.zero_grad()

        self.split_counter += 1
        if self.split_counter == 0:
            self.view.auto_split()    # 切分参数
            self.view.reinitialize()  # 清空梯度
            trainer.reinitialize()    # 把所有参数加入优化器

    def load_dataset(self, split, epoch=1, **kwargs):
        super(AutoShareTranslationTask, self).load_dataset(split, epoch)
        if split == 'train':
            old_dataset: RoundRobinZipDatasets = self.datasets[split]
            self.datasets[split] = FastRoundRobinDataset(datasets=old_dataset.datasets, eval_key=old_dataset.eval_key)

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        if not isinstance(dataset, FastRoundRobinDataset):
            # 不是训练集数据
            return super(AutoShareTranslationTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache)

        if dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        assert isinstance(dataset, FastRoundRobinDataset)
        logger.info("get indices ordered by example size")
        with data_utils.numpy_seed(seed):
            indices = {
                key: value.ordered_indices()
                for key, value in dataset.datasets.items()
            }
        logger.info("filter examples that are too large")
        indices = {
            key: self.filter_indices_by_size(indices[key], value, max_positions[key], ignore_invalid_inputs)
            for key, value in dataset.datasets.items()
        }
        logger.info("create mini-batches with given size constraints")
        batch_sampler = {
            key: value.batch_by_size(
                indices[key],
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
            for key, value in dataset.datasets.items()
        }
        logger.info("return a reusable, sharded iterator")
        epoch_iter = {
            key: iterators.EpochBatchIterator(
                dataset=value,
                collate_fn=value.collater,
                batch_sampler=batch_sampler[key],
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=0,
                epoch=epoch,
                buffer_size=data_buffer_size,
            )
            for key, value in dataset.datasets.items()
        }

        assert self.sample_method in ['temperature', 'uniform']
        if self.sample_method == 'temperature':
            sizes = {
                key: len(epoch_iter[key]) ** (1 / self.sample_temperature)
                for key in epoch_iter.keys()
            }
            sample_prop = {
                key: sizes[key] / max(sizes.values())
                for key in self.dataset('train').datasets.keys()
            }
        else:
            sample_prop = {key: 1 for key in self.dataset('train').datasets.keys()}

        self.dataset_to_epoch_iter[dataset] = MyEpochBatchIterator(epoch_iter, sample_prop)
        return self.dataset_to_epoch_iter[dataset]

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        models = [models[0].models[self.lang_pairs[0]]]
        return super(AutoShareTranslationTask, self).build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs)


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
