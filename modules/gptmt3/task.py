import os
import torch
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig
from fairseq.data.monolingual_dataset import MonolingualDataset


def get_lang_tok_index(vocab):
    return int(vocab.indices['<<<<1>>>>'])


class SrcMaskMonolingualDataset(MonolingualDataset):
    def __getitem__(self, index):
        source, future_target, past_target = self.dataset[index]
        source, target = self._make_source_target(source, future_target, past_target)
        source, target = self._maybe_add_bos(source, target)
        target[:torch.nonzero(target == get_lang_tok_index(self.vocab))+1] = self.vocab.pad()
        return {"id": index, "source": source, "target": target}


@register_task("gpt_lm", dataclass=LanguageModelingConfig)
class LanguageModelingGptTask(LanguageModelingTask):

    def _initialize_dataset(self, **kwargs):
        return SrcMaskMonolingualDataset(**kwargs)

    def build_model(self, args):
        if 'test' in self.datasets:
            args.model_parallel_size = 1
            args.arch = 'zhgpt_2_infer'
        model = super(LanguageModelingGptTask, self).build_model(args)
        if 'test' not in self.datasets:
            gpt_model_path = os.environ.get('GPT_MODEL', None)
            if gpt_model_path is None:
                raise NotImplementedError('gpt model env var not set')
            states = torch.load(gpt_model_path.format(args.distributed_rank % 2))
            model_dict = states['model']
            model.load_state_dict(model_dict)
        return model

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        # batch size = 1
        src_tokens = sample['net_input']['src_tokens'][0]
        prefix = src_tokens[:torch.nonzero(src_tokens == get_lang_tok_index(self.source_dictionary))+1]
        prefix_tokens = prefix.unsqueeze(0)
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.args, "add_bos_token", False):
                bos_token = self.source_dictionary.bos()
            else:
                bos_token = self.source_dictionary.eos()

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
            if prefix_tokens[:, 0].eq(bos_token).all():
                prefix_tokens = prefix_tokens[:, 1:]

            result = generator.generate(models, sample, prefix_tokens=prefix_tokens, bos_token=bos_token)
            return result
