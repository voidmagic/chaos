import logging
import math
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("cluster_task")
class ClusterTask(TranslationMultiSimpleEpochTask):
    train_iterator = None
    last_valid_sample, last_valid_key, last_valid_loss, last_train_key = None, None, None, None
    random_state_share, random_state_indiv = None, None
    affinity_interval = 5  # calculate every N steps

    def gen_random_train_batch(self):
        if self.random_state_indiv is None or self.random_state_share is None:
            self.random_state_indiv = random.Random(dist.get_rank() if dist.is_initialized() else 0)
            self.random_state_share = random.Random(0)

        if self.train_iterator is None:
            self.train_iterator = []
            for key, dataset in zip(self.datasets["train"].keys, self.datasets["train"].datasets):
                batch_iterator = self.get_batch_iterator(
                    dataset=dataset, max_tokens=self.args.max_tokens_valid,
                    max_sentences=self.args.batch_size_valid,
                    max_positions=self.max_positions(),
                    ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                    required_batch_size_multiple=self.args.required_batch_size_multiple,
                    seed=dist.get_rank() if dist.is_initialized() else 0, num_workers=self.args.num_workers,
                    data_buffer_size=self.args.data_buffer_size).next_epoch_itr()
                self.train_iterator.append((key, batch_iterator))
        # 在计算affinity的时候，不同的GPU先选择同样的语言，再选择不同的batch
        key, iterator = self.random_state_share.choice(self.train_iterator)
        return key, next(iterator)

    @torch.no_grad()
    def calculate_valid_loss(self, model, criterion):
        # 随机搞一个batch，计算其loss
        model.eval()
        _, _, logging_output = criterion(model, utils.move_to_cuda(self.last_valid_sample))
        loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()
        model.train()
        return loss

    def save_to_affinity(self, loss_before, loss_after):
        if dist.is_initialized():
            # gather last loss
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_before)
            loss_befores = output

            # gather this loss
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_after)
            loss_afters = output
        else:
            # single gpu training
            loss_befores = [loss_before]
            loss_afters = [loss_after]

        for loss_before, loss_after in zip(loss_befores, loss_afters):
            logger.info("Affinity | {} | {} | {} | {}".format(self.last_train_key, self.last_valid_key, float(loss_before), float(loss_after)))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if update_num > 0 and update_num % self.affinity_interval == 0:
            self.last_valid_key, self.last_valid_sample = self.gen_random_train_batch()
            self.last_valid_loss = self.calculate_valid_loss(model, criterion)
            self.last_train_key, train_sample = self.gen_random_train_batch()
            loss, sample_size, logging_output = super(ClusterTask, self).train_step(
                utils.move_to_cuda(train_sample), model, criterion, optimizer, update_num, ignore_grad)
            return loss, sample_size, logging_output

        if self.last_train_key is not None:
            # 上一步使用self.last_train_key对应的batch更新
            loss = self.calculate_valid_loss(model, criterion)
            self.save_to_affinity(self.last_valid_loss, loss)
            self.last_train_key = None

        return super(ClusterTask, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
