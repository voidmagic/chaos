import os

from fairseq.data import indexed_dataset, data_utils, PrependTokenDataset, LanguagePairDataset, ConcatDataset
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks import register_task
from .utils.dataset import MultiParallelDataset


@register_task("sync_mnmt")
class SyncTranslationTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,en-es')

    def load_dataset(self, split, **kwargs):
        if getattr(self.args, 'gen_subset', 'test') == split:
            pass
            # src, tgt = self.args.source_lang, self.args.target_lang
            #
            # if indexed_dataset.dataset_exists(
            #         os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
            #     prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            # elif indexed_dataset.dataset_exists(
            #         os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
            #     prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            # else:
            #     raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))
            #
            # src_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
            # tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)
            # src_prepend_dataset = PrependTokenDataset(
            #     src_raw_dataset, self.src_dict.index('__2<{}>__'.format(self.args.target_lang)))
            # self.datasets[split] = LanguagePairDataset(
            #     src_prepend_dataset, src_prepend_dataset.sizes, self.src_dict,
            #     tgt_raw_dataset, tgt_raw_dataset.sizes, self.tgt_dict)
            # return

        src_datasets = []
        tgt_datasets = []
        for lang_pair in self.args.lang_pairs:
            src, tgt = lang_pair
            if indexed_dataset.dataset_exists(
                    os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(
                    os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.src_dict)
            tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.tgt_dict)
            tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, self.tgt_dict.index('__2<{}>__'.format(tgt)))

            src_datasets.append(src_raw_dataset)
            tgt_datasets.append(tgt_prepend_dataset)

        self.datasets[split] = MultiParallelDataset(src_datasets, tgt_datasets, self.src_dict)


    @classmethod
    def setup_task(cls, args, **kwargs):
        args.lang_pairs = args.lang_pairs.split(',')
        args.lang_pairs = [lang_pair.split('-') for lang_pair in args.lang_pairs]
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = args.lang_pairs[0]
        task = super(SyncTranslationTask, cls).setup_task(args)
        langs = list(set([lang for pair in args.lang_pairs for lang in pair]))
        for lang_token in sorted(['__2<{}>__'.format(lang) for lang in langs]):
            task.src_dict.add_symbol(lang_token)
            task.tgt_dict.add_symbol(lang_token)
        return task

