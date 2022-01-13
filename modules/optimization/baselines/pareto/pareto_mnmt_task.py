"""
Support:
    pareto MTL with min norm solver.
    freeze part of the model, currently last decoder layer.
"""

import collections
import logging

import torch
from fairseq import utils
from fairseq.tasks import register_task
from torch.nn.utils import parameters_to_vector
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from modules.optimization.baselines.pareto.min_norm_solver import find_min_norm_element

logger = logging.getLogger(__name__)


@register_task("pareto_mnmt")
class ParetoMultilingualNeuralMachineTranslationTask(TranslationMultiSimpleEpochTask):
    def __init__(self, args, langs, dicts, training):
        super(ParetoMultilingualNeuralMachineTranslationTask, self).__init__(args, langs, dicts, training)
        self.alpha = None
        self.smooth = args.smooth_alpha
        self.last_update_or_start = args.start_update_alpha
        self.update_interval = args.update_interval_alpha
        self.eval_param_key = args.eval_key_alpha

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--smooth-alpha', default=0.1, type=float, help='smooth alpha to avoid zero weight.')
        parser.add_argument('--start-update-alpha', default=10000, type=int, help='step to start update alpha')
        parser.add_argument('--update-interval-alpha', default=200, type=int, help='alpha update interval')
        parser.add_argument('--eval-key-alpha', default=None, type=str, help='use param with name of eval key to eval alpha')
        TranslationMultiSimpleEpochTask.add_args(parser)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if update_num > self.last_update_or_start and update_num % self.update_interval == 0:
            self.reset_alpha(model, criterion)
            self.last_update_or_start = update_num

        model.train()
        model.set_num_updates(update_num)

        loss, sample_size, logging_output = criterion(model, sample, reduce=False)

        if ignore_grad:
            loss *= 0

        if self.alpha is not None:
            task_ids = self.infer_task(sample)
            weights = torch.tensor([self.alpha[task_id] for task_id in task_ids]).view(-1, 1)
            loss = loss.view(len(task_ids), -1) * weights.to(loss.device)
        optimizer.backward(loss.sum())

        return loss, sample_size, {
            key: value.sum() if isinstance(value, torch.Tensor) else value for key, value in logging_output.items()
        }

    def build_model(self, args):
        model = super(ParetoMultilingualNeuralMachineTranslationTask, self).build_model(args)
        for name, param in model.named_parameters():
            if self.eval_param_key is not None and not name.startswith(self.eval_param_key):
                param.requires_grad = False
        return model

    def reset_alpha(self, model, criterion):

        gradient_for_each_task = collections.defaultdict(float)
        model.eval()  # disable dropout

        # calculate jacobians for each task
        for dataset in self.dataset(self.args.valid_subset).datasets:
            batch_iterator = self.get_batch_iterator(
                dataset=dataset,
                max_tokens=self.args.max_tokens_valid,
                max_positions=utils.resolve_max_positions(self.max_positions(), model.max_positions()),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                seed=self.args.seed,
            ).next_epoch_itr(shuffle=False)

            model.zero_grad()
            task_id = dataset.src.token
            for sample in batch_iterator:
                sample = utils.move_to_cuda(sample)
                loss, _, _ = criterion(model, sample)
                loss = loss / len(batch_iterator)
                loss.backward()
            gradients = []
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    gradients.append(p.grad)
            gradient_for_each_task[task_id] = parameters_to_vector(gradients)

        model.train()  # enable dropout
        model.zero_grad()

        gradient_for_each_task_sorted = sorted(gradient_for_each_task.items(), key=lambda item: item[0])
        gradient_for_each_task_tensor = torch.stack([item[1] for item in gradient_for_each_task_sorted])
        try:
            alpha, _ = find_min_norm_element(gradient_for_each_task_tensor.float(), max_iter=250)
            self.alpha = {item[0]: max(a, self.smooth) * len(alpha) for item, a in zip(gradient_for_each_task_sorted, alpha)}
            logger.info(f"Reset alpha: {self.alpha}")
        except UnboundLocalError as _:
            logger.info(f"Reset alpha filed.")

    def infer_task(self, sample):
        source_tokens = sample["net_input"]["src_tokens"]
        idx = torch.arange(source_tokens.shape[1], 0, -1).to(source_tokens.device)
        indices = torch.argmax((source_tokens - self.source_dictionary.pad_index) * idx, dim=1, keepdim=True)
        task_ids = torch.gather(source_tokens, dim=1, index=indices)
        return task_ids.view(-1).tolist()
