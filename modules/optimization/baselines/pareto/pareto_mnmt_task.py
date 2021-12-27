import collections
import logging

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from modules.optimization.baselines.pareto.min_norm_solver import find_min_norm_element

logger = logging.getLogger(__name__)


@register_task("pareto_mnmt")
class ParetoMultilingualNeuralMachineTranslationTask(TranslationMultiSimpleEpochTask):
    alpha, smooth = None, 0.1
    last_update_or_start, update_interval = 10000, 200

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

    def reset_alpha(self, model, criterion):

        gradient_for_each_task = collections.defaultdict(float)

        # calculate jacobians for each task
        for dataset in self.dataset(self.args.valid_subset).datasets:
            batch_iterator = self.get_batch_iterator(
                dataset=dataset,
                max_tokens=self.args.max_tokens_valid,
                max_positions=utils.resolve_max_positions(self.max_positions(), model.max_positions()),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                seed=self.args.seed,
            ).next_epoch_itr(shuffle=False)

            model.eval()  # disable dropout
            for sample in batch_iterator:
                sample = utils.move_to_cuda(sample)
                task_id = self.infer_task(sample)
                assert len(set(task_id)) == 1  # only contains one task
                model.zero_grad()
                # 计算这个batch对应的loss
                loss, _, _ = criterion(model, sample)
                loss.backward()
                gradients = []
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        gradients.append(p.grad.view(-1).data.cpu())
                gradient_for_each_task[task_id[0]] += torch.cat(gradients) / len(batch_iterator)
                model.zero_grad()
            model.train()  # enable dropout

        gradient_for_each_task_sorted = sorted(gradient_for_each_task.items(), key=lambda item: item[0])
        gradient_for_each_task_tensor = torch.stack([item[1] for item in gradient_for_each_task_sorted])
        try:
            alpha, _ = find_min_norm_element(gradient_for_each_task_tensor.float(), max_iter=250)
            self.alpha = {item[0]: max(a * len(alpha), self.smooth) for item, a in zip(gradient_for_each_task_sorted, alpha)}
            logger.info(f"Reset alpha: {self.alpha}")
        except UnboundLocalError as _:
            logger.info(f"Reset alpha filed.")

    def infer_task(self, sample):
        source_tokens = sample["net_input"]["src_tokens"]
        idx = torch.arange(source_tokens.shape[1], 0, -1).to(source_tokens.device)
        indices = torch.argmax((source_tokens - self.source_dictionary.pad_index) * idx, dim=1, keepdim=True)
        task_ids = torch.gather(source_tokens, dim=1, index=indices)
        return task_ids.view(-1).tolist()
