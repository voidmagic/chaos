import os

import torch
from fairseq import utils
from fairseq.data import (
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingTask, LanguageModelingConfig, dataclass, field, SAMPLE_BREAK_MODE_CHOICES

from .utils.dataset import BilingualDatasetFromMonolingual, MonolingualSrcMaskDataset, MonolingualDataset


@dataclass
class GptMtConfig(LanguageModelingConfig):
    only_target: bool = field(default=False, metadata={"help": "only use target loss"})

    # 测试的时候指定源语言和目标语言
    source_language: str = field(default='', metadata={"help": "source language"})
    target_language: str = field(default='', metadata={"help": "target language"})


@register_task("gpt_mt", dataclass=GptMtConfig)
class GPTForMtTask(LanguageModelingTask):

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        if split == 'test':
            # 在测试的时候，保证一个句子作为一个sample
            self.args.sample_break_mode = SAMPLE_BREAK_MODE_CHOICES('eos')
            assert self.args.source_language != '' and self.args.target_language != ''

        source_idx = target_idx = None
        if self.args.source_language != '':
            source_idx = self.dictionary.index('<<{}>>'.format(self.args.source_language))
            target_idx = self.dictionary.index('<<{}>>'.format(self.args.target_language))
            assert source_idx != self.dictionary.unk_index and target_idx != self.dictionary.unk_index

        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        data_path = paths[(epoch - 1) % len(paths)]
        split_path = os.path.join(data_path, split)

        dataset = data_utils.load_indexed_dataset(split_path, self.dictionary, self.args.dataset_impl, combine=combine)

        dataset = maybe_shorten_dataset(
            dataset,
            split,
            self.args.shorten_data_split_list,
            self.args.shorten_method,
            self.args.tokens_per_sample,
            self.args.seed,
        )

        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample,
            pad=self.dictionary.pad(),
            eos=self.dictionary.eos(),
            break_mode=self.args.sample_break_mode,
            include_targets=True,
        )

        add_eos_for_other_targets = (self.args.sample_break_mode is not None and self.args.sample_break_mode != "none")

        self.datasets[split] = self._initialize_dataset(
            source_idx=source_idx,  # only used for test
            target_idx=target_idx,  # only used for test
            split=split,  # only used for test
            dataset=dataset,
            sizes=dataset.sizes,
            src_vocab=self.dictionary,
            tgt_vocab=self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets,
            shuffle=True,
            targets=self.targets,
            add_bos_token=self.args.add_bos_token,
        )

    def _initialize_dataset(self, split, source_idx, target_idx, **kwargs):
        if split == 'test':
            return BilingualDatasetFromMonolingual(source_idx=source_idx, target_idx=target_idx, **kwargs)
        elif self.args.only_target:
            return MonolingualSrcMaskDataset(**kwargs)
        else:
            return MonolingualDataset(**kwargs)

    def filter_indices_by_size(self, indices, dataset, max_positions=None, ignore_invalid_inputs=False):
        return super(GPTForMtTask, self).filter_indices_by_size(indices, dataset, max_positions=max_positions, ignore_invalid_inputs=True)

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        results = super(GPTForMtTask, self).inference_step(generator, models, sample, prefix_tokens, constraints)
        target_idx = self.dictionary.index('<{}>'.format(self.args.target_language))

        for result in results:
            for beam in result:
                idx = torch.nonzero(beam['tokens'] == target_idx, as_tuple=False) + 1
                beam['tokens'] = beam['tokens'][idx[-1]:]
        return results
