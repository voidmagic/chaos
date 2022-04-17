import logging
import math
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.criterions import cross_entropy
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


# train with mix sample, calculate affinity with single language

@register_task("affinity_task_3")
class AffinityTask3(TranslationMultiSimpleEpochTask):
    affinities = []
    valid_batches, train_batches = None, None
    last_valid_sample, last_valid_key, last_valid_loss = None, None, None
    last_train_sample, last_train_key, last_train_loss = None, None, None
    random_state_share, random_state_indiv = None, None
    affinity_interval = 20  # calculate every N steps

    def gen_random_train_batch(self):
        if self.random_state_indiv is None or self.random_state_share is None:
            self.random_state_indiv = random.Random(dist.get_rank() if dist.is_initialized() else 0)
            self.random_state_share = random.Random(0)

        if self.train_batches is None:
            self.train_batches = []
            datasets, _ = self.data_manager.load_split_datasets("valid", True)
            for valid_key, dataset in datasets:
                lang_train_batches = []
                old_method, self.args.sampling_method = self.args.sampling_method, "RoundRobin"
                batch_iterator = self.get_batch_iterator(
                    dataset=dataset, max_tokens=self.args.max_tokens_valid,
                    max_sentences=self.args.batch_size_valid,
                    max_positions=self.max_positions(),
                    ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                    required_batch_size_multiple=self.args.required_batch_size_multiple,
                    seed=self.args.seed,
                    num_workers=self.args.num_workers,
                    data_buffer_size=self.args.data_buffer_size).next_epoch_itr()
                self.args.sampling_method = old_method
                for sample in batch_iterator:
                    lang_train_batches.append([valid_key, sample])
                self.train_batches.append(lang_train_batches)
        # 在计算affinity的时候，不同的GPU先选择同样的语言，再选择不同的batch
        return self.random_state_indiv.choice(self.random_state_share.choice(self.train_batches))

    def get_random_valid_batch(self):
        if self.random_state_indiv is None or self.random_state_share is None:
            self.random_state_indiv = random.Random(dist.get_rank() if dist.is_initialized() else 0)
            self.random_state_share = random.Random(0)

        if self.valid_batches is None:
            self.valid_batches = []
            datasets, _ = self.data_manager.load_split_datasets("valid", True)
            for valid_key, dataset in datasets:
                lang_validation_batches = []
                old_method, self.args.sampling_method = self.args.sampling_method, "RoundRobin"
                batch_iterator = self.get_batch_iterator(
                    dataset=dataset, max_tokens=self.args.max_tokens_valid,
                    max_sentences=self.args.batch_size_valid,
                    max_positions=self.max_positions(),
                    ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                    required_batch_size_multiple=self.args.required_batch_size_multiple,
                    seed=self.args.seed,
                    num_workers=self.args.num_workers,
                    data_buffer_size=self.args.data_buffer_size).next_epoch_itr()
                self.args.sampling_method = old_method
                for sample in batch_iterator:
                    lang_validation_batches.append([valid_key, sample])
                self.valid_batches.append(lang_validation_batches)
        return self.random_state_indiv.choice(self.random_state_indiv.choice(self.valid_batches))

    @torch.no_grad()
    def calculate_valid_loss(self, model, criterion):
        # 随机搞一个batch，计算其loss
        model.eval()
        self.last_valid_key, self.last_valid_sample = self.get_random_valid_batch()
        _, _, logging_output = criterion(model, utils.move_to_cuda(self.last_valid_sample))
        self.last_valid_loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone() / math.log(2)

    @torch.no_grad()
    def calculate_affinity(self, model, criterion):
        if self.last_valid_loss is not None:  # 上一步成功计算了！
            model.eval()
            # 重新计算一下valid loss，上次计算loss的batch，计算基于last_sample参数更新后的loss变化
            _, _, logging_output = criterion(model, utils.move_to_cuda(self.last_valid_sample))
            loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone() / math.log(2)
            loss_diff_valid = (1 - loss / self.last_valid_loss).cpu().data.clone()
            # 保存为一个affinity
            self.save_to_affinity(self.last_valid_loss, loss, loss_diff_valid)

    def save_to_affinity(self, loss_before, loss_after, loss_diff):
        if dist.is_initialized():
            # gather validation key
            output = ["" for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_valid_key)
            last_valid_keys = output

            # gather last loss
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_before)
            loss_befores = output

            # gather this loss
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_after)
            loss_afters = output

            # gather loss diff
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_diff)
            loss_diffs = output
        else:
            # single gpu training
            last_valid_keys = [self.last_valid_key]
            loss_befores = [loss_before]
            loss_afters = [loss_after]
            loss_diffs = [loss_diff]

        for last_valid_key, loss_before, loss_after, loss_diff in zip(last_valid_keys, loss_befores, loss_afters, loss_diffs):
            logger.info("Affinity | " + self.last_train_key + " | " + "None" + " | " + last_valid_key + " | " + str(float(loss_before)) + " | " + str(float(loss_after)) + " | " + str(float(loss_diff)))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if update_num % self.affinity_interval == 0:
            self.calculate_valid_loss(model, criterion)
            self.last_train_key, self.last_train_sample = self.gen_random_train_batch()
            loss, sample_size, logging_output = super(AffinityTask3, self).train_step(utils.move_to_cuda(self.last_train_sample), model, criterion, optimizer, update_num, ignore_grad)
            return loss, sample_size, logging_output

        if update_num % self.affinity_interval == 1:
            self.calculate_affinity(model, criterion)

        return super(AffinityTask3, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
