import logging
import os
import time

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.trainer import Trainer
from modules.sample_mnmt.task import SampledMultilingualTask
from modules.auto_share.tasks.view import ModelView

logger = logging.getLogger(__name__)


@register_task('auto_share')
class AutoShareTranslationTask(SampledMultilingualTask):

    @staticmethod
    def add_args(parser):
        SampledMultilingualTask.add_args(parser)
        parser.add_argument('--split-only-record', default='False', type=str, metavar='BOOL')
        parser.add_argument('--split-interval', default=5, type=int)
        parser.add_argument('--split-subset', default='multi', type=str)
        parser.add_argument('--split-all', default='True', type=str, metavar='BOOL')
        parser.add_argument('--split-threshold', default=0.0, type=float)
        parser.add_argument('--split-granularity', default='parameter', choices=['parameter', 'module', 'layer'])


    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.view: ModelView = None
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.split_counter, self.split_interval = 0, args.split_interval
        self.split_only_record = utils.eval_bool(args.split_only_record)
        self.load_dataset(args.split_subset)

    def build_model(self, args):
        model = super(AutoShareTranslationTask, self).build_model(args)
        self.view = ModelView(model, args=args)
        return model

    def begin_valid_epoch(self, epoch, model):
        trainer = get_trainer()
        criterion = trainer.criterion

        logger.info("Start accumulating gradient")
        dataset_for_split = self.dataset(self.args.split_subset)

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
                if sample.get(lang_pair, None) is None:
                    continue
                loss, _, _ = criterion(model.models[lang_pair], sample[lang_pair])
                # 缩放一下，避免出现NAN
                loss = loss / len(batch_iterator) / self.split_interval
                loss.backward()
                self.view.accum_gradient(lang_pair)
                model.zero_grad()
        model.train()

        self.split_counter += 1
        if self.split_counter % self.split_interval == 0:
            if self.split_only_record:
                torch.save(self.view.gradients, os.path.join(self.args.save_dir, "{}.pt".format(int(time.time()))))
            else:
                self.view.auto_split()      # 切分参数
                trainer.reinitialize()      # 把所有参数加入优化器
                logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))

            self.view.clear_gradient()  # 清空梯度


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
