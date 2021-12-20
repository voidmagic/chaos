import os

from fairseq.data import data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from .dataset import Dataset

from .generator import SequenceGenerator


@register_task('simple_translation')
class SimpleMachineTranslationTask(TranslationTask):
    def load_dataset(self, split, **kwargs):
        if self.args.source_lang and self.args.target_lang:
            src, tgt = self.args.source_lang, self.args.target_lang
        else:
            src, tgt = data_utils.infer_language_pair(self.args.data)
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)
        self.datasets[split] = Dataset(src_dataset, src_dataset.sizes, self.src_dict, tgt_dataset, tgt_dataset.sizes, self.tgt_dict)

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None):
        return super(SimpleMachineTranslationTask, self).build_generator(models, args, SequenceGenerator)
