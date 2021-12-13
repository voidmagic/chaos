import logging
from itertools import chain

import torch
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
from fairseq.trainer import Trainer
from fairseq.criterions import cross_entropy
from modules.param_diff.view import ModelView

logger = logging.getLogger(__name__)


@register_task('parameter_differentiation_task')
class ParameterDifferentiationTask(MultilingualTranslationTask):
    view: ModelView = None

    def build_model(self, args):
        model = super(ParameterDifferentiationTask, self).build_model(args)
        self.view = ModelView(model)
        return model

    def record_gradient(self, model):
        logger.info("Start accumulating gradient")

        criterion = cross_entropy.CrossEntropyCriterion(task=self, sentence_avg=False)
        batch_iterator = self.get_batch_iterator(
            dataset=self.dataset('valid'),
            max_tokens=self.args.max_tokens_valid,
            max_sentences=self.args.batch_size_valid,
            max_positions=utils.resolve_max_positions(self.max_positions(), model.max_positions()),
        ).next_epoch_itr(shuffle=False)

        model.eval()
        for i, sample in enumerate(batch_iterator):
            sample = utils.move_to_cuda(sample)
            for lang_pair in self.lang_pairs:
                model.zero_grad()
                if sample is None or sample.get(lang_pair, None) is None:
                    continue
                loss, _, _ = criterion(model.models[lang_pair], sample[lang_pair])
                loss = loss / len(batch_iterator)
                loss.backward()
                self.view.accum_gradient(lang_pair)
                model.zero_grad()
        model.train()

    def begin_valid_epoch(self, epoch, model):
        trainer = get_trainer()
        old_state = trainer.optimizer.state_dict()
        exp_avg_dict, exp_avg_sq_dict = record_optimizer_state(old_state, trainer)
        self.record_gradient(model)
        logger.info("num. model params before: {}".format(sum(p.numel() for p in model.parameters())))
        name_mapping = list(self.view.auto_split())
        reload_optimizer_state(trainer, exp_avg_dict, exp_avg_sq_dict, name_mapping, old_state)
        logger.info("num. model params after: {}".format(sum(p.numel() for p in model.parameters())))
        self.view.clear_gradient()


def record_optimizer_state(state, trainer):
    exp_avg_dict, exp_avg_sq_dict, offset = {}, {}, 0
    all_named_params = chain(trainer.model.named_parameters(), trainer.criterion.named_parameters())
    for name, param in list(filter(lambda p: p[1].requires_grad, all_named_params)):
        exp_avg_dict[name] = state['state'][0]['exp_avg'][offset: offset + param.numel()]
        exp_avg_sq_dict[name] = state['state'][0]['exp_avg_sq'][offset: offset + param.numel()]
        offset += param.numel()
    return exp_avg_dict, exp_avg_sq_dict


def reload_optimizer_state(trainer, exp_avg_dict, exp_avg_sq_dict, name_mapping, state):
    trainer._optimizer = None
    exp_avg_new, exp_avg_sq_new = [], []
    all_named_params = chain(trainer.model.named_parameters(), trainer.criterion.named_parameters())
    for name, param in list(filter(lambda p: p[1].requires_grad, all_named_params)):
        if name not in exp_avg_dict:
            assert name not in exp_avg_sq_dict
            for new_name, old_name in name_mapping:
                name = name.replace(new_name, old_name)
        exp_avg_new.append(exp_avg_dict[name])
        exp_avg_sq_new.append(exp_avg_sq_dict[name])

    state['state'][0]['exp_avg'] = torch.cat(exp_avg_new)
    state['state'][0]['exp_avg_sq'] = torch.cat(exp_avg_sq_new)
    trainer.optimizer.load_state_dict(state)


def get_trainer() -> Trainer:
    import gc
    for obj in gc.get_objects():
        if isinstance(obj, Trainer):
            return obj
