import os

from fairseq.data import data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from .dataset import Dataset


@register_task('simple_translation')
class SimpleMachineTranslationTask(TranslationTask):
    def load_dataset(self, split, **kwargs):
        src, tgt = data_utils.infer_language_pair(self.args.data)
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)
        self.datasets[split] = Dataset(src_dataset, src_dataset.sizes, self.src_dict, tgt_dataset, tgt_dataset.sizes, self.tgt_dict)
