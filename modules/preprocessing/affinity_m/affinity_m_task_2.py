import collections
import logging
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

logger = logging.getLogger(__name__)


@register_task("affinitym_task_2")
class AffinityMTask2(MultilingualTranslationTask):
    validation_batches = None
    affinities = []
    last_valid_sample, last_valid_key, last_valid_loss = None, None, None
    last_train_sample, last_train_loss, last_train_key = None, None, None
    random_state_share, random_state_indiv = None, None

    def get_random_batch(self):
        if self.validation_batches is None:
            self.random_state_indiv = random.Random(dist.get_rank() if dist.is_initialized() else 0)
            self.random_state_share = random.Random(0)
            self.validation_batches = collections.defaultdict(list)
            batch_iterator = self.get_batch_iterator(
                dataset=self.datasets["valid"], max_tokens=self.args.max_tokens_valid,
                max_sentences=self.args.batch_size_valid,
                max_positions=self.max_positions(),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                seed=self.args.seed,
                num_workers=self.args.num_workers,
                data_buffer_size=self.args.data_buffer_size).next_epoch_itr()
            for sample in batch_iterator:
                for valid_key, valid_sample in sample.items():
                    self.validation_batches[valid_key].append(utils.move_to_cuda(valid_sample))
        valid_key = self.random_state_indiv.choice(list(self.validation_batches.keys()))
        return valid_key, self.random_state_indiv.choice(self.validation_batches[valid_key])

    @torch.no_grad()
    def calculate_affinity(self, model, criterion):
        model.eval()
        if self.last_valid_sample is not None:
            # 上次计算loss的batch，计算基于last_sample参数更新后的loss变化
            _, _, logging_output = criterion(model, self.last_valid_sample)
            loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()
            loss_diff_valid = (1 - loss / self.last_valid_loss).cpu().data.clone()
            # 保存为一个affinity
            self.save_to_affinity(self.last_valid_loss, loss, loss_diff_valid)

        # 随机搞一个batch，计算其loss
        self.last_valid_key, self.last_valid_sample = self.get_random_batch()
        _, _, logging_output = criterion(model, self.last_valid_sample)
        self.last_valid_loss = (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()

    def save_to_affinity(self, loss_before, loss_after, loss_diff):
        if dist.is_initialized():
            # multi gpu training, gather training sample IDs
            output = [torch.zeros(0) for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_train_sample["id"].cpu())
            instance_ids = torch.cat(output, dim=0).tolist()

            # gather validation key
            output = ["" for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_valid_key)
            last_valid_keys = output

            # gather train key
            output = ["" for _ in range(dist.get_world_size())]
            dist.all_gather_object(output, self.last_train_key)
            last_train_keys = output

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
            instance_ids = self.last_train_sample["id"].cpu().tolist()
            last_valid_keys = [self.last_valid_key]
            last_train_keys = [self.last_train_key]
            loss_befores = [loss_before]
            loss_afters = [loss_after]
            loss_diffs = [loss_diff]

        for last_train_key, last_valid_key, loss_before, loss_after, loss_diff in zip(last_train_keys, last_valid_keys, loss_befores, loss_afters, loss_diffs):
            logger.info("Affinity | " + last_train_key + " | " + str(instance_ids) + " | " + last_valid_key + " | " + str(float(loss_before)) + " | " + str(float(loss_after)) + " | " + str(float(loss_diff)))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.calculate_affinity(model, criterion)
        # 选择一个语言训练，其他的语言为None
        self.last_train_key = self.random_state_share.choice(list(sample.keys()))
        sample = {key: value if key == self.last_train_key else None for key, value in sample.items()}
        loss, sample_size, logging_output = super(AffinityMTask2, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        self.last_train_sample, self.last_train_loss = sample[self.last_train_key], (logging_output["loss"] / logging_output["ntokens"]).cpu().data.clone()
        return loss, sample_size, logging_output

    def _per_lang_pair_train_loss(self, lang_pair, model, update_num, criterion, sample, optimizer, ignore_grad):
        loss, sample_size, logging_output = criterion(model, sample[lang_pair])
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def _per_lang_pair_valid_loss(self, lang_pair, model, criterion, sample):
        return criterion(model, sample[lang_pair])

    def build_model(self, args, from_checkpoint=False):
        from fairseq import models
        model = models.build_model(args, self, from_checkpoint)
        model.max_positions = lambda: None
        return model

