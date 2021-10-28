import os

from fairseq.data import data_utils, ConcatDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data.language_pair_dataset import LanguagePairDataset


@register_task('bi_translation')
class BiDirMachineTranslationTask(TranslationTask):
    def load_dataset(self, split, **kwargs):
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)

        if getattr(self.args, 'train_subset', 'train') == split:
            src_dataset, tgt_dataset = ConcatDataset([src_dataset, tgt_dataset]), ConcatDataset([tgt_dataset, src_dataset])

        self.datasets[split] = LanguagePairDataset(src_dataset, src_dataset.sizes, self.src_dict, tgt_dataset, tgt_dataset.sizes, self.tgt_dict)
