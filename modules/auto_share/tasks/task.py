import logging
import os
import time
from itertools import chain

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.trainer import Trainer
from fairseq.criterions import cross_entropy
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
        parser.add_argument('--split-accum', default=5, type=int)
        parser.add_argument('--split-start', default=0, type=int)
        parser.add_argument('--split-subset', default='multi', type=str)
        parser.add_argument('--split-count', default=1, type=int, metavar='BOOL', help='每次拆分多少个')
        parser.add_argument('--split-granularity', default='parameter', choices=['parameter', 'module', 'layer'])
        parser.add_argument('--split-momentum', default=0.0, type=float, help='其他任务的动量')


    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.view: ModelView = None
        self.cuda = torch.cuda.is_available() and not args.cpu
        self.split_counter, self.split_interval = 0, args.split_interval
        self.split_accum = min(args.split_accum, args.split_interval)
        self.split_only_record = utils.eval_bool(args.split_only_record)
        self.load_dataset(args.split_subset)
        self.split_momentum = args.split_momentum
        self.last_sample = None

    def build_model(self, args):
        model = super(AutoShareTranslationTask, self).build_model(args)
        self.view = ModelView(model, args=args)
        return model

    def record_gradient(self, model):
        logger.info("Start accumulating gradient")

        # 使用交叉熵，不使用label smoothing
        criterion = cross_entropy.CrossEntropyCriterion(task=self, sentence_avg=False)

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
            for lang_pair in self.lang_pairs:  # 正常情况下，每个batch只有一个语言
                model.zero_grad()
                if sample is None or sample.get(lang_pair, None) is None:
                    continue
                # 计算这个batch对应的loss
                loss, _, _ = criterion(model.models[lang_pair], sample[lang_pair])
                # 缩放一下，避免出现NAN
                loss = loss / len(batch_iterator) / self.split_accum
                loss.backward()
                self.view.accum_gradient(lang_pair)
                model.zero_grad()
        model.train()

    def begin_valid_epoch(self, epoch, model):
        self.split_counter += 1
        if self.split_counter < self.args.split_start:
            return
        if self.split_interval - (self.split_counter - 1) % self.split_interval > self.split_accum:
            return

        trainer = get_trainer()

        old_state = trainer.optimizer.state_dict()
        exp_avg_dict, exp_avg_sq_dict = record_optimizer_state(old_state, trainer)

        # 记录梯度
        self.record_gradient(model)

        if self.split_counter % self.split_interval == 0:
            if self.split_only_record:
                # 仅记录梯度，不拆分
                torch.save(self.view.gradients, os.path.join(self.args.save_dir, "{}.pt".format(int(time.time()))))
            else:
                # 切分参数
                logger.info("num. model params before: {}".format(sum(p.numel() for p in model.parameters())))
                name_mapping = list(self.view.auto_split())
                reload_optimizer_state(trainer, exp_avg_dict, exp_avg_sq_dict, name_mapping, old_state)
                logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))
            self.view.clear_gradient()  # 清空梯度

    # def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
    #     # 上一步的样本，用当前语言的模型进行训练，梯度进行动量缩放
    #     if self.last_sample is not None and self.split_momentum != 0.0:
    #         loss, _, _ = criterion(model.models[lang_pair], self.last_sample)
    #         loss *= self.split_momentum
    #         optimizer.backward(loss)
    #     self.last_sample = sample[lang_pair]
    #
    #     return super(AutoShareTranslationTask, self)._per_lang_pair_train_loss(
    #         lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad)


def record_optimizer_state(state, trainer):
    # 记录Adam的状态
    exp_avg_dict, exp_avg_sq_dict, offset = {}, {}, 0
    all_named_params = chain(trainer.model.named_parameters(), trainer.criterion.named_parameters())
    for name, param in list(filter(lambda p: p[1].requires_grad, all_named_params)):
        exp_avg_dict[name] = state['state'][0]['exp_avg'][offset: offset + param.numel()]
        exp_avg_sq_dict[name] = state['state'][0]['exp_avg_sq'][offset: offset + param.numel()]
        offset += param.numel()
    return exp_avg_dict, exp_avg_sq_dict


def reload_optimizer_state(trainer, exp_avg_dict, exp_avg_sq_dict, name_mapping, state):
    # 把所有参数加入优化器，并保留原Adam优化器的状态
    # 1. 删除原来的Adam
    trainer._optimizer = None
    # 2. 修改原Adam的状态
    exp_avg_new, exp_avg_sq_new = [], []
    all_named_params = chain(trainer.model.named_parameters(), trainer.criterion.named_parameters())
    for name, param in list(filter(lambda p: p[1].requires_grad, all_named_params)):
        # 如果参数以前就有，那就用以前的Adam状态
        # 如果参数是新的，根据新旧参数映射词典改名
        if name not in exp_avg_dict:
            assert name not in exp_avg_sq_dict
            for new_name, old_name in name_mapping:
                name = name.replace(new_name, old_name)
        exp_avg_new.append(exp_avg_dict[name])
        exp_avg_sq_new.append(exp_avg_sq_dict[name])

    state['state'][0]['exp_avg'] = torch.cat(exp_avg_new)
    state['state'][0]['exp_avg_sq'] = torch.cat(exp_avg_sq_new)
    trainer.optimizer.load_state_dict(state)


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
