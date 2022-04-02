import collections
import logging
import random

import torch
import torch.distributed as dist
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

logger = logging.getLogger(__name__)


@register_task("msam")
class MultiSamTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):
    validation_batches = None
    closures = []
    samples = []
    criterion = None
    grad_state = collections.defaultdict(dict)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--sam-adaptive', default='False', type=str)
        parser.add_argument('--sam-rho', default=0.05, type=float)
        TranslationMultiSimpleEpochTask.add_args(parser)

    def get_random_batch(self):
        if self.validation_batches is None:
            random.seed(dist.get_rank() if dist.is_initialized() else 0)
            self.validation_batches = []
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
                    lang_validation_batches.append(utils.move_to_cuda(sample))
                self.validation_batches.append(lang_validation_batches)
        return random.choice(random.choice(self.validation_batches))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        adaptive = self.args.sam_adaptive == "True"
        sam_rho = self.args.sam_rho

        def step(the_sample):
            model.train()
            model.set_num_updates(update_num)
            loss, sample_size, logging_output = criterion(model, the_sample)
            optimizer.backward(loss)
            return loss, sample_size, logging_output

        try:
            # first step, get gradient
            step(self.get_random_batch())
            with torch.no_grad():
                grad_norm = optimizer.clip_grad_norm(max_norm=0.0)
                scale = sam_rho / (grad_norm + 1e-12)
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    self.grad_state[p]["old_p"] = p.data.clone()
                    e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
                    p.add_(e_w)  # climb to the local maximum "w + e(w)"
                optimizer.zero_grad()

            loss, sample_size, logging_output = step(sample)

            with torch.no_grad():
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    p.data = self.grad_state[p]["old_p"]  # get back to "w" from "w + e(w)"
        except OverflowError:
            optimizer.zero_grad()
            loss, sample_size, logging_output = super(MultiSamTranslationMultiSimpleEpochTask, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        return loss, sample_size, logging_output
