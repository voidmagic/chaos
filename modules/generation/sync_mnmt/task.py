import os

from fairseq.data import indexed_dataset, data_utils, PrependTokenDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask

from modules.basics.google_mnmt.google_mnmt_task import GoogleMultilingualTranslationTask
from .utils.dataset import MultiParallelDataset
from .utils.generator import SequenceGenerator


class Config:
    n_lang = None
    manner = 'tanh'
    tanh_weight = 0.1
    lang_idx = set()
    infer_target = None
    non_proj = False
    weight_record = []
    weight_path = None
    current_layer = -1
    fusion_last = False


@register_task("sync_mnmt")
class SyncTranslationTask(TranslationMultiSimpleEpochTask):

    @classmethod
    def add_args(cls, parser):
        TranslationMultiSimpleEpochTask.add_args(parser)
        parser.add_argument('--manner', default='tanh', type=str)
        parser.add_argument('--tanh-weight', default=0.1, type=float)
        parser.add_argument('--non-proj', action='store_true')
        parser.add_argument('--weight-path', default=None, type=str)
        parser.add_argument('--fusion-last', action='store_true')
        parser.add_argument('--train-cla', action='store_true')

    @classmethod
    def setup_task(cls, args, **kwargs):
        task = super(SyncTranslationTask, cls).setup_task(args)
        Config.n_lang = len(args.lang_pairs)
        Config.manner = args.manner
        Config.tanh_weight = args.tanh_weight
        Config.non_proj = args.non_proj
        Config.weight_path = args.weight_path
        Config.fusion_last = args.fusion_last
        return task

    def load_dataset(self, split, **kwargs):

        def load_data(src, tgt):
            if indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, src, tgt, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
            elif indexed_dataset.dataset_exists(os.path.join(self.args.data, '{}.{}-{}.{}'.format(split, tgt, src, src)), None):
                prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, tgt, src))
            else:
                raise FileNotFoundError(os.path.join(self.args.data, '{}.{}-{}.*'.format(split, tgt, src)))

            src_raw_dataset = data_utils.load_indexed_dataset(prefix + src, self.source_dictionary)
            tgt_raw_dataset = data_utils.load_indexed_dataset(prefix + tgt, self.target_dictionary)
            tgt_prepend_dataset = PrependTokenDataset(tgt_raw_dataset, self.source_dictionary.index('__{}__'.format(tgt)))
            return src_raw_dataset, tgt_prepend_dataset

        if split == 'test' and self.args.source_lang != self.args.target_lang:
            src_dataset, tgt_dataset = load_data(self.args.source_lang, self.args.target_lang)
            self.datasets[split] = MultiParallelDataset(src_dataset, [tgt_dataset], self.source_dictionary)
            Config.infer_target = self.source_dictionary.index('__{}__'.format(self.args.target_lang))
            for _, tgt in self.args.lang_pairs:
                Config.lang_idx.add(self.source_dictionary.index('__{}__'.format(tgt)))
            return

        src_datasets = []
        tgt_datasets = []
        for lang_pair in self.args.lang_pairs:
            src, tgt = lang_pair.split('-')
            src_dataset, tgt_dataset = load_data(src, tgt)
            src_datasets.append(src_dataset)
            tgt_datasets.append(tgt_dataset)

        self.datasets[split] = MultiParallelDataset(src_datasets[0], tgt_datasets, self.source_dictionary)

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, extra=None):
        return super(SyncTranslationTask, self).build_generator(models, args, SequenceGenerator, extra_gen_cls_kwargs)

