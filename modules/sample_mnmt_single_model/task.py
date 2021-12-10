import os
from collections import OrderedDict

from fairseq.data import indexed_dataset, data_utils, PrependTokenDataset, LanguagePairDataset
from fairseq.tasks import register_task

from modules.sample_mnmt.dataset import MultilingualSampledDataset
from modules.google_mnmt.google_mnmt_task import GoogleMultilingualTranslationTask


@register_task("sample_mnmt_share")
class SampledMultilingualSingleModelTask(GoogleMultilingualTranslationTask):
    @staticmethod
    def add_args(parser):
        GoogleMultilingualTranslationTask.add_args(parser)
        parser.add_argument('--sample-method', default='proportional', choices=['temperature', 'proportional', 'uniform'])
        parser.add_argument('--sample-temperature', default=5, type=int)

    def load_dataset(self, split, epoch=1, **kwargs):
        super(SampledMultilingualSingleModelTask, self).load_dataset(split, **kwargs)
        if split != 'train' and split != 'valid':
            return

        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict)
            return (PrependTokenDataset(src_raw_dataset, self.src_dict.index('__2<{}>__'.format(tgt))),
                    data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict))

        datasets = []
        for lang_pair in self.args.lang_pairs:
            src_prepend_dataset, tgt_raw_dataset = load_data(lang_pair[0], lang_pair[1])
            datasets.append((
                '-'.join(lang_pair),
                LanguagePairDataset(src_prepend_dataset, src_prepend_dataset.sizes, self.src_dict,
                                          tgt_raw_dataset, tgt_raw_dataset.sizes, self.tgt_dict)
            ))

        self.datasets[split] = MultilingualSampledDataset(
            OrderedDict(datasets),
            sample_method=self.args.sample_method,
            temperature=self.args.sample_temperature
        )


    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        for key, value in sample.items():
            if value is None: continue
            return super(SampledMultilingualSingleModelTask, self).train_step(value, model, criterion, optimizer, update_num, ignore_grad)
        return None

    def valid_step(self, sample, model, criterion):
        for key, value in sample.items():
            if value is None: continue
            return super(SampledMultilingualSingleModelTask, self).valid_step(value, model, criterion)
        return None
