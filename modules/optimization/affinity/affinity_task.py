import logging
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.criterions import cross_entropy
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("affinity_task")
class AffinityTask(TranslationMultiSimpleEpochTask):
    validation_batches = None
    affinities = []
    last_valid_sample, last_key, last_valid_loss = None, None, None
    last_train_sample, last_train_loss = None, None

    def get_random_batch(self):
        if self.validation_batches is None:
            random.seed(0)
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
                    self.validation_batches.append([valid_key, utils.move_to_cuda(sample)])
        return random.choice(self.validation_batches)

    @torch.no_grad()
    def calculate_affinity(self, model, criterion):
        model.eval()
        if self.last_valid_sample is not None:
            # 上次计算loss的batch，计算基于last_sample参数更新后的loss变化
            _, _, logging_output = criterion(model, self.last_valid_sample)
            loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()
            # 保存为一个affinity
            self.save_to_affinity(self.last_valid_loss, loss)

        # 随机搞一个batch，计算其loss
        self.last_key, self.last_valid_sample = self.get_random_batch()
        _, _, logging_output = criterion(model, self.last_valid_sample)
        self.last_valid_loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()

    def save_to_affinity(self, loss_before, loss_after):
        if dist.is_initialized():
            # multi gpu training, gather training sample IDs
            output = [torch.zeros(0) for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_train_sample["id"].cpu())
            instance_ids = torch.cat(output, dim=0).tolist()

            # gather training sample languages
            output = [torch.zeros(0) for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_train_sample["net_input"]["src_tokens"][:, 0].cpu())
            language_ids = torch.cat(output, dim=0).tolist()

            # gather validation key
            output = ["" for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_key)
            last_keys = output

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
            instance_ids = self.last_train_sample["id"].cpu().tolist()
            language_ids = self.last_train_sample["net_input"]["src_tokens"][:, 0].cpu().tolist()
            last_keys = [self.last_key]
            loss_befores = [loss_before]
            loss_afters = [loss_after]

        language_strs = self.source_dictionary.string(language_ids).split()
        assert len(language_strs) == len(instance_ids)
        for last_key, loss_before, loss_after in zip(last_keys, loss_befores, loss_afters):
            logger.info("Affinity | " + str(language_strs) + " | " + str(instance_ids) + " | " + last_key + " | " + str(float(loss_before)) + " | " + str(float(loss_after)))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.calculate_affinity(model, criterion)
        loss, sample_size, logging_output = super(AffinityTask, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        self.last_train_sample, self.last_train_loss = sample, (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()
        return loss, sample_size, logging_output
