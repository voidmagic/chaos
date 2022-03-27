import logging
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("affinity_task")
class AffinityTask(TranslationMultiSimpleEpochTask):
    validation_batches = None
    affinities = []
    last_batch, last_key, last_loss = None, None, None
    last_sample = None

    def get_random_batch(self):
        if self.validation_batches is None:
            self.validation_batches = []
            datasets, _ = self.data_manager.load_split_datasets("valid", True)
            for valid_key, dataset in datasets:
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
                    self.validation_batches.append([valid_key, sample])
        return random.choice(self.validation_batches)

    @torch.no_grad()
    def calculate_affinity(self, model, criterion):
        if self.last_batch is not None:
            # 上次计算loss的batch，计算基于last_sample参数更新后的loss变化
            loss, _, _ = criterion(model, utils.move_to_cuda(self.last_batch))
            loss_diff = (1 - loss / self.last_loss).cpu().data.clone()
            # 保存为一个affinity
            self.save_to_affinity(loss_diff)

        # 随机搞一个batch，计算其loss
        self.last_key, self.last_batch = self.get_random_batch()
        self.last_loss, _, _ = criterion(model, utils.move_to_cuda(self.last_batch))
        self.last_loss = self.last_loss.data.clone()

    def save_to_affinity(self, loss_diff):
        if dist.is_initialized():
            # multi gpu training
            output = [torch.zeros(0) for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_sample["id"].cpu())
            instance_ids = torch.cat(output, dim=0).tolist()
            output = [torch.zeros(0) for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_sample["net_input"]["src_tokens"][:, 0].cpu())
            language_ids = torch.cat(output, dim=0).tolist()
            output = ["" for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_key)
            last_keys = output
            output = [0.0 for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, loss_diff)
            loss_diffs = output
        else:
            # single gpu training
            instance_ids = self.last_sample["id"].cpu().tolist()
            language_ids = self.last_sample["net_input"]["src_tokens"][:, 0].cpu().tolist()
            last_keys = [self.last_key]
            loss_diffs = [loss_diff]

        language_strs = self.source_dictionary.string(language_ids).split()
        assert len(language_strs) == len(instance_ids)
        for last_key, loss_diff in zip(last_keys, loss_diffs):
            logger.info("Affinity | " + str(language_strs) + " | " + str(instance_ids) + " | " + last_key + " | " + str(float(loss_diff)))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.calculate_affinity(model, criterion)
        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        self.last_sample = sample
        return loss, sample_size, logging_output
