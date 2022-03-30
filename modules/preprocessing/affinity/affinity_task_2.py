import logging
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.criterions import cross_entropy
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("affinity_task_2")
class AffinityTask2(TranslationMultiSimpleEpochTask):
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
        if self.last_train_sample is not None:
            # 上次计算loss的batch，计算基于last_sample参数更新后的loss变化
            loss, _, logging_output = criterion(model, self.last_train_sample, reduce=False)
            loss = loss.view(logging_output["nsentences"], -1).mean(dim=1).cpu().data.clone()
            loss_diff_train = (1 - loss / self.last_train_loss).cpu().data.clone()
            # 保存为一个affinity
            self.save_to_affinity(loss_diff_train)

    def save_to_affinity(self, loss_diff):
        instance_ids = self.last_train_sample["id"].cpu().tolist()
        language_ids = self.last_train_sample["net_input"]["src_tokens"][:, 0].cpu().tolist()
        language_strs = self.source_dictionary.string(language_ids).split()
        logger.info("Affinity | " + " ".join(language_strs) + " | " + " ".join(map(str, instance_ids)) + " | " + " ".join(map(str, loss_diff.tolist())))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self.calculate_affinity(model, criterion)

        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample, reduce=False)
        if ignore_grad: loss *= 0
        optimizer.backward(loss.sum())

        self.last_train_sample, self.last_train_loss = sample, loss.view(logging_output["nsentences"], -1).mean(dim=1).cpu().data.clone()
        return loss.sum(), sample_size, {key: value.sum() if isinstance(value, torch.Tensor) else value for key, value in logging_output.items()}
