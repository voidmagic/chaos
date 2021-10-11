import os

import torch
from fairseq.data import indexed_dataset, data_utils, PrependTokenDataset, ConcatDataset, LanguagePairDataset
from fairseq.tasks import register_task

from modules.google_mnmt.google_mnmt_task import GoogleMultilingualTranslationTask
from .utils.dataset import MultiParallelDataset


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


@register_task("sync_mnmt")
class SyncTranslationTask(GoogleMultilingualTranslationTask):

    def load_dataset(self, split, **kwargs):
        Config.n_lang = len(self.args.lang_pairs)

        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(
                    os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict)
            tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict)
            tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, self.src_dict.index('__2<{}>__'.format(tgt)))
            return src_raw_dataset, tgt_prepend_dataset

        src_datasets = []
        tgt_datasets = []
        for lang_pair in self.args.lang_pairs:
            src_dataset, tgt_dataset = load_data(lang_pair[0], lang_pair[1])
            src_datasets.append(src_dataset)
            tgt_datasets.append(tgt_dataset)

        assert dataset_equal(*src_datasets)
        self.datasets[split] = MultiParallelDataset(src_datasets[0], tgt_datasets, self.src_dict)


