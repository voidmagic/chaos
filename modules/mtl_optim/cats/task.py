import collections
from typing import List

import torch
import torch.optim as optim
from fairseq.tasks import register_task

from modules.sample_mnmt_single_model.task import SampledMultilingualSingleModelTask
from modules.mtl_optim.pareto.utils import get_d_paretomtl
from fairseq import utils
from modules.mtl_optim.cats.model import ModelParetoWeightLambda


@register_task("cats_mnmt")
class CatsLearningMultilingualTask(SampledMultilingualSingleModelTask):
    weight = dict()
    lang_pairs = list()
    weight_model = None
    weight_optimizer = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        task: CatsLearningMultilingualTask = super(CatsLearningMultilingualTask, cls).setup_task(args, **kwargs)
        task.lang_pairs = ["-".join(pair) for pair in args.lang_pairs]
        task.weight_model = ModelParetoWeightLambda(len(task.lang_pairs)).cuda()
        task.weight_optimizer = optim.SGD(task.weight_model.parameters(), lr=0.00001)
        return task

    def update_weight(self, model, criterion):
        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset('valid'),
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
        model.zero_grad()
        grads_dict = collections.defaultdict(list)
        losses_dict = collections.defaultdict(list)
        for i, sample in enumerate(batch_iterator):
            sample = utils.move_to_cuda(sample)

            for key, value in sample.items():
                if value is None: continue

                loss, _, _ = criterion(model, value)
                loss.backward()
                grads_dict[key].append(torch.cat([p.grad.clone().detach().flatten() for p in model.parameters() if p.grad is not None]))
                losses_dict[key].append(loss.clone().detach())
                model.zero_grad()

        grads_dict_tensor = dict()
        losses_dict_tensor = dict()
        for key in self.lang_pairs:
            grads_dict_tensor[key] = torch.mean(torch.stack(grads_dict[key], dim=0), dim=0)
            losses_dict_tensor[key] = torch.mean(torch.stack(losses_dict[key], dim=0), dim=0)

        grads = torch.stack([grads_dict_tensor[key] for key in self.lang_pairs], dim=0)
        losses = torch.stack([losses_dict_tensor[key] for key in self.lang_pairs], dim=0)
        self.weight_model.zero_grad()
        weight_loss = self.weight_model(losses, grads)
        weight_loss.backward()
        self.weight_optimizer.step()
        del weight_loss
        model.train()
        self.weight = dict(zip(self.lang_pairs, self.weight_model.alpha.tolist()))

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if update_num % 100 == 0 and update_num > 40000:
            self.update_weight(model, criterion)

        loss = sample_size = logging_output = None
        for key, value in sample.items():
            if value is None: continue

            model.train()
            model.set_num_updates(update_num)
            loss, sample_size, logging_output = criterion(model, value)
            loss *= self.weight.get(key, 1)
            optimizer.backward(loss)
        return loss, sample_size, logging_output

