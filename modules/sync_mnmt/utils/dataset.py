import logging

import numpy as np
import torch
from fairseq.data import data_utils
from fairseq.data.language_pair_dataset import LanguagePairDataset as FairseqLanguagePairDataset

logger = logging.getLogger(__name__)



def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(tokens, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            tokens,
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_multiple=pad_to_multiple,
        )

    n_lang = len(samples[0]['target'])

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge([s["source"] for s in samples], left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target_tokens = [s["target"][i] for i in range(n_lang) for s in samples]
    target = merge(target_tokens, left_pad=left_pad_target)
    target = torch.stack(torch.chunk(target, len(samples[0]['target'])))
    target = target.index_select(1, sort_order)

    tgt_lengths = torch.LongTensor([[s["target"][i].ne(pad_idx).long().sum() for s in samples] for i in range(n_lang)])
    tgt_lengths = tgt_lengths.index_select(1, sort_order)
    ntokens = tgt_lengths.sum().item()

    prev_output_tokens = merge(target_tokens, left_pad=left_pad_target, move_eos_to_beginning=True)
    prev_output_tokens = torch.stack(torch.chunk(prev_output_tokens, n_lang))
    prev_output_tokens = prev_output_tokens.index_select(1, sort_order)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "prev_output_tokens": prev_output_tokens.view(n_lang * len(samples), -1)
        },
        "target": target.view(n_lang * len(samples), -1),
    }
    return batch



class MultiParallelDataset(FairseqLanguagePairDataset):
    def __init__(self, src, tgt, src_dict, *args, **kwargs):
        super(MultiParallelDataset, self).__init__(src, src.sizes, src_dict, *args, **kwargs)
        self.tgt = tgt
        self.tgt_sizes = np.sum(np.array([d.sizes for d in tgt]), axis=0)
        self.tgt_sizes_l = [d.sizes for d in tgt]
        self.src_dict = src_dict
        self.tgt_dict = src_dict

    def __getitem__(self, index):
        tgt_item = [tgt[index] for tgt in self.tgt]
        src_item = self.src[index]
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], sum(s[index] for s in self.tgt_sizes_l))

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            return indices[np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")]
