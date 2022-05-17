import os
from collections import OrderedDict

from fairseq.data import PrependTokenDataset, data_utils, indexed_dataset, LanguagePairDataset, RoundRobinZipDatasets
from fairseq.tasks import register_task

from modules.basics.sample_mnmt.dataset import MultilingualSampledDataset
from modules.basics.sample_mnmt.task import SampledMultilingualTask


@register_task('adapter_task')
class XDAdapterTask(SampledMultilingualTask):

    def load_dataset(self, split, epoch=1, **kwargs):
        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.dicts[src])
            return (PrependTokenDataset(src_raw_dataset, self.dicts[src].index('__{}__'.format(tgt))),
                    data_utils.load_indexed_dataset(prefix + tgt, self.dicts[tgt]))

        def load_language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            src_prepend_dataset, tgt_raw_dataset = load_data(src, tgt)
            dataset = LanguagePairDataset(src_prepend_dataset, src_prepend_dataset.sizes, self.dicts[src], tgt_raw_dataset, tgt_raw_dataset.sizes, self.dicts[tgt])
            return dataset

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict(
                [
                    (lang_pair, load_language_pair_dataset(lang_pair))
                    for lang_pair in self.lang_pairs
                ]
            ),
            eval_key=None
            if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )
        if split != 'train':
            return
        self.datasets[split] = MultilingualSampledDataset(
            self.datasets[split].datasets,
            self.datasets[split].eval_key,
            sample_method=self.args.sample_method,
            temperature=self.args.sample_temperature
        )


