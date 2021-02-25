import os

from fairseq.data import data_utils, LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from modules.gptmt2.tasks.dataset import Dataset


@register_task("lm_mt_task")
class GPTMTTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--reverse-source', action='store_true', default=False)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)
        data_class = Dataset if getattr(self.args, 'reverse_source', False) else LanguagePairDataset
        self.datasets[split] = data_class(src_dataset, src_dataset.sizes, self.src_dict,
                                          tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                                          left_pad_source=self.args.left_pad_source)
