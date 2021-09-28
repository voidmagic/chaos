import collections
import copy
import random
import numpy as np
from collections import defaultdict
import torch
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

@register_task("pcgrad_mnmt")
class PcGradMultilingualTask(MultilingualTranslationTask):

    @staticmethod
    def add_args(parser):
        MultilingualTranslationTask.add_args(parser)
        parser.add_argument('--target-lang-pair', default=None)

    def __init__(self, args, dicts, training):
        super(PcGradMultilingualTask, self).__init__(args, dicts, training)
        self.target_lang_pair = None

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
        curr_lang_pairs = [lang_pair for lang_pair in self.model_lang_pairs if sample[lang_pair] is not None and len(sample[lang_pair]) != 0]
        objectives = {}
        for idx, lang_pair in enumerate(curr_lang_pairs):
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
            if ignore_grad:
                loss *= 0
            objectives[lang_pair] = loss
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{lang_pair}:{k}"] += logging_output[k]
        self.pc_backward2(objectives, optimizer, model.models)
        return agg_loss, agg_sample_size, agg_logging_output

    def pc_backward1(self, objectives, optimizer, models):
        all_grads = collections.defaultdict(dict)
        all_names = set()
        for lang_pair, obj in objectives.items():
            optimizer.backward(obj)
            for name, param in models[lang_pair].named_parameters():
                all_grads[lang_pair][name] = param.grad.clone()
                all_names.add(name)
            models.zero_grad()

        pc_grad = copy.deepcopy(all_grads)
        for name in all_names:
            if 'layer' not in name: continue
            for lang_pair_i in all_grads.keys():
                g_i = pc_grad[lang_pair_i][name]
                shape, dtype = g_i.shape, g_i.dtype
                g_i = g_i.flatten().float()
                lang_pairs = list(all_grads.keys())
                random.shuffle(lang_pairs)
                for lang_pair_j in lang_pairs:
                    g_j = all_grads[lang_pair_j][name].flatten().float()
                    g_i_g_j = torch.dot(g_i, g_j)
                    if g_i_g_j < 0:
                        g_i -= g_i_g_j * g_j / (g_j.norm() ** 2)
                pc_grad[lang_pair_i][name] = g_i.view(shape).type(dtype)

        for lang_pair in all_grads.keys():
            for name, param in models[lang_pair].named_parameters():
                param.grad = pc_grad[lang_pair][name] if param.grad is None else param.grad + pc_grad[lang_pair][name]

    def pc_backward2(self, objectives, optimizer, models):
        lang_pairs = [k for k, _ in objectives.items()]
        objectives = [v for _, v in objectives.items()]
        grads, shapes, has_grads = self._pack_grad(objectives, optimizer)
        pc_grad = self._project_conflicting(grads, has_grads, lang_pairs)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad, optimizer)
        return

    def _project_conflicting(self, grads, has_grads, lang_pairs):
        if self.target_lang_pair not in lang_pairs:
            return torch.stack([g for g in grads]).sum(dim=0)
        target_index = lang_pairs.index(self.target_lang_pair)
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for idx, g_i in enumerate(pc_grad):
            dtype = g_i.dtype
            g_i = g_i.float()
            g_j = grads[target_index].float()
            g_i_g_j = torch.dot(g_i, g_j)
            if g_i_g_j < 0:
                g_i -= g_i_g_j * g_j / (g_j.norm() ** 2)
            pc_grad[idx] = g_i.type(dtype)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads, optimizer):
        idx = 0
        for p in optimizer.fp16_params:
            p.grad = grads[idx]
            idx += 1
        return

    def _pack_grad(self, objectives, optimizer):
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            optimizer.zero_grad()
            optimizer.backward(obj)
            grad, shape, has_grad = self._retrieve_grad(optimizer)
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, optimizer):
        grad, shape, has_grad = [], [], []
        for p in optimizer.fp16_params:
            if p.grad is None:
                shape.append(p.shape)
                grad.append(torch.zeros_like(p).to(p.device))
                has_grad.append(torch.zeros_like(p).to(p.device))
                continue
            shape.append(p.grad.shape)
            grad.append(p.grad.clone())
            has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
