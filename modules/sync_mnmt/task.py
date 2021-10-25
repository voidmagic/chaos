import os

import torch
from fairseq.data import indexed_dataset, data_utils, PrependTokenDataset, ConcatDataset, LanguagePairDataset
from fairseq.tasks import register_task

from modules.google_mnmt.google_mnmt_task import GoogleMultilingualTranslationTask
from .utils.dataset import MultiParallelDataset
from .utils.generator import SequenceGenerator
from fairseq import metrics, search, tokenizer, utils


def dataset_equal(*datasets):
    if len(set([len(d) for d in datasets])) > 1:
        return False

    first = datasets[0]
    for i in range(len(first)):
        for dataset in datasets:
            if not torch.equal(first[i], dataset[i]):
                return False
    return True


class Config:
    n_lang = None
    manner = 'tanh'
    tanh_weight = 0.1
    lang_idx = set()
    infer_target = None


@register_task("sync_mnmt")
class SyncTranslationTask(GoogleMultilingualTranslationTask):

    def load_dataset(self, split, **kwargs):
        Config.n_lang = len(self.args.lang_pairs)

        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict)
            tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict)
            tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, self.src_dict.index('__2<{}>__'.format(tgt)))
            return src_raw_dataset, tgt_prepend_dataset


        if split == 'test':
            src_dataset, tgt_dataset = load_data(self.args.source_lang, self.args.target_lang)
            self.datasets[split] = MultiParallelDataset(src_dataset, [tgt_dataset], self.src_dict)
            Config.infer_target = self.src_dict.index('__2<{}>__'.format(self.args.target_lang))
            for _, tgt in self.args.lang_pairs:
                Config.lang_idx.add(self.src_dict.index('__2<{}>__'.format(tgt)))
            return

        src_datasets = []
        tgt_datasets = []
        for lang_pair in self.args.lang_pairs:
            src_dataset, tgt_dataset = load_data(lang_pair[0], lang_pair[1])
            src_datasets.append(src_dataset)
            tgt_datasets.append(tgt_dataset)

        assert dataset_equal(*src_datasets)
        self.datasets[split] = MultiParallelDataset(src_datasets[0], tgt_datasets, self.src_dict)

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        return super(SyncTranslationTask, self).build_generator(models, args, SequenceGenerator, extra_gen_cls_kwargs)
