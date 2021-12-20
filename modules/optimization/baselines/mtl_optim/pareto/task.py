import torch
import numpy as np
from fairseq.tasks import register_task

from modules.basics.sample_mnmt_single_model.task import SampledMultilingualSingleModelTask
from modules.optimization.baselines.mtl_optim.pareto.utils import get_d_paretomtl
from fairseq import utils


def circle_points(n):
    t = np.linspace(0, 0.5 * np.pi, n)
    x = np.cos(t)
    y = np.sin(t)
    ref_vec = torch.tensor(np.c_[x, y]).cuda().float()
    return ref_vec


@register_task("pareto_mnmt")
class ParetoLearningMultilingualTask(SampledMultilingualSingleModelTask):
    weight = dict()
    lang_pairs = None
    preference = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        task: ParetoLearningMultilingualTask = super(ParetoLearningMultilingualTask, cls).setup_task(args, **kwargs)
        task.lang_pairs = ["-".join(pair) for pair in args.lang_pairs]
        task.preference = circle_points(5)
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

        losses_vec = dict()
        grads = dict()
        model.eval()
        model.zero_grad()
        for i, sample in enumerate(batch_iterator):
            sample = utils.move_to_cuda(sample)

            for key, value in sample.items():
                if value is None or key in grads: continue
                loss, _, _ = criterion(model, value)
                losses_vec[key] = loss.data
                loss.backward()
                grads[key] = [p.grad.clone().detach().flatten() for p in model.parameters() if p.grad is not None]
                model.zero_grad()

        grads = torch.stack([torch.cat(grads[key]) for key in self.lang_pairs]).float()
        losses_vec = torch.stack([losses_vec[key] for key in self.lang_pairs])
        weight_vec = get_d_paretomtl(grads, losses_vec, self.preference, 0)
        normalize_coeff = len(self.lang_pairs) / torch.sum(torch.abs(weight_vec))
        weight_vec = weight_vec * normalize_coeff
        self.weight = dict(zip(self.lang_pairs, weight_vec.tolist()))
        model.train()

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if update_num % 10 == 0 and update_num > 15000:
        # if update_num % 10 == 0 and update_num > 20:
            try:
                self.update_weight(model, criterion)
            except UnboundLocalError as e:
                pass

        loss = sample_size = logging_output = None
        for key, value in sample.items():
            if value is None: continue

            model.train()
            model.set_num_updates(update_num)
            loss, sample_size, logging_output = criterion(model, value)
            loss *= self.weight.get(key, 1)
            optimizer.backward(loss)
        return loss, sample_size, logging_output

