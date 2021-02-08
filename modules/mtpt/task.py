import os
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.data import data_utils, LanguagePairDataset
from fairseq import utils
from .utils.mixed_dataset import MixedLanguagePairDataset


@register_task('translation_pretrain')
class MachineTranslationPreTrainTask(TranslationTask):

    def load_dataset(self, split, **kwargs):
        shared_dict = self.src_dict
        paths = utils.split_paths(self.args.data)
        bilingual_path = paths[0]
        src, tgt = data_utils.infer_language_pair(bilingual_path)
        prefix = os.path.join(bilingual_path, '{}.{}-{}.'.format(split, src, tgt))

        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, shared_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, shared_dict)

        bilingual_dataset = LanguagePairDataset(src_dataset, src_dataset.sizes, shared_dict, tgt_dataset, tgt_dataset.sizes, shared_dict)

        if len(paths) == 1 or split != 'train':
            self.datasets[split] = bilingual_dataset
            return

        assert len(paths) == 2
        monolingual_path = paths[1]
        src, tgt = data_utils.infer_language_pair(monolingual_path)
        prefix = os.path.join(monolingual_path, '{}.{}-{}.'.format(split, src, tgt))
        src_mono_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, shared_dict)
        tgt_mono_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, shared_dict)
        monolingual_dataset = LanguagePairDataset(src_mono_dataset, src_mono_dataset.sizes, shared_dict, tgt_mono_dataset, tgt_mono_dataset.sizes, shared_dict)

        self.datasets[split] = MixedLanguagePairDataset(bilingual_dataset=bilingual_dataset, monolingual_dataset=monolingual_dataset)
