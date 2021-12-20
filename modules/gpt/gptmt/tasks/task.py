import logging
import os

import torch
from fairseq.data import data_utils, LanguagePairDataset
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from modules.gpt.gptmt.models.share_transformer import ShareEncoderDecoderTransformerModel
from .dataset import Dataset, FakeLanguagePairDataset


logger = logging.getLogger(__name__)


@register_task("lm_mt_task")
class GPTMTTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        parser.add_argument('--reverse-source', action='store_true', default=False)
        # 预训练数据保存的路径
        parser.add_argument('--pre-train-source', default=None, metavar='SRC')
        parser.add_argument('--pre-train-target', default=None, metavar='SRC')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        self.datasets[split] = self.load_bilingual_dataset(split, epoch, combine, **kwargs)

        if self.args.pre_train_target is not None and split == 'train':
            tgt_mono_dataset = self.load_monolingual_dataset(self.args.pre_train_target, is_source=True)
            self.datasets[split] = tgt_mono_dataset

    def load_bilingual_dataset(self, split, epoch=1, combine=False, **kwargs):
        src, tgt = self.args.source_lang, self.args.target_lang
        prefix = os.path.join(self.args.data, '{}.{}-{}.'.format(split, src, tgt))
        src_dataset = data_utils.load_indexed_dataset(prefix + self.args.source_lang, self.src_dict)
        tgt_dataset = data_utils.load_indexed_dataset(prefix + self.args.target_lang, self.tgt_dict)
        data_class = Dataset if getattr(self.args, 'reverse_source', False) else LanguagePairDataset
        return data_class(src_dataset, src_dataset.sizes, self.src_dict, tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                          left_pad_source=self.args.left_pad_source)

    def load_monolingual_dataset(self, path, is_source=False, is_target=False):
        # 文件格式：train.bin train.idx
        dataset = data_utils.load_indexed_dataset(os.path.join(path, 'train'), self.src_dict)
        dataset = FakeLanguagePairDataset(dataset, dataset.sizes, self.src_dict,
                                          left_pad_source=self.args.left_pad_source)
        return dataset

    def build_model(self, args):
        model: ShareEncoderDecoderTransformerModel = super(GPTMTTask, self).build_model(args)
        pretrain_path = os.environ.get('PRETRAIN', None)
        if pretrain_path is not None:
            logger.info('Load pretrain model from {}'.format(pretrain_path))
            stat = torch.load(pretrain_path)['model']
            model.load_state_dict({})
        return model

    def build_generator(self, *args, **kwargs):
        generator = super(GPTMTTask, self).build_generator(*args, **kwargs)
        generator.search.stop_on_max_len = True
        return generator
