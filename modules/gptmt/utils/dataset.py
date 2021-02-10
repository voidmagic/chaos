import numpy as np
import torch
from fairseq.data.monolingual_dataset import MonolingualDataset as FairseqMonolingualDataset


def get_language_token_indexes(vocab, tokens):
    # get number of special tokens
    symbols = vocab.symbols
    lang_indexes = []
    for i, symbol in enumerate(symbols):
        if symbol.startswith('<<'):  # language tokens
            if i in tokens:
                lang_indexes.append(torch.nonzero(torch.tensor(tokens == i)))
        elif symbol.startswith('<'):  # fairseq special tokens
            pass
        else:
            break
    return min(lang_indexes), max(lang_indexes)


class BilingualDatasetFromMonolingual(FairseqMonolingualDataset):
    def __init__(self, source_idx, target_idx, **kwargs):
        super(BilingualDatasetFromMonolingual, self).__init__(**kwargs)
        self.filter(source_idx, target_idx)

    def filter(self, source_idx, target_idx):
        # 提供过滤的id（把合并的双向双语的测试集过滤为单向）
        assert hasattr(self, "sizes") and isinstance(self.sizes, np.ndarray)
        for index in range(len(self)):
            tokens = self.dataset.dataset[index]
            if source_idx not in tokens and target_idx not in tokens:
                # other language pair in dataset
                self.sizes[index] = 99999
            elif torch.nonzero(tokens.eq(source_idx)) > torch.nonzero(tokens.eq(target_idx)):
                # target to source
                self.sizes[index] = 99999

    def __getitem__(self, index):
        # 只在测试的时候使用，source作为前缀，包含target是目标端句子
        source, _, _ = self.dataset[index]
        lang_index_src, lang_index_tgt = get_language_token_indexes(self.vocab, source)
        source, target = source[:lang_index_tgt+1], source[lang_index_tgt+1:]
        return {'id': index, 'source': source, 'target': target}


class MonolingualSrcMaskDataset(FairseqMonolingualDataset):
    def __getitem__(self, index):
        # 把返回词典中的target部分（用来计算loss）中的前半部分（源语言部分）设为pad，不计算其loss
        source, future_target, past_target = self.dataset[index]
        source, target = self._make_source_target(source, future_target, past_target)
        source, target = self._maybe_add_bos(source, target)
        lang_index_src, lang_index_tgt = get_language_token_indexes(self.vocab, target)
        target[:lang_index_tgt + 1] = self.vocab.pad()
        return {"id": index, "source": source, "target": target}


class MonolingualDataset(FairseqMonolingualDataset):
    def __getitem__(self, index):
        # 把target的语言标签设置为pad，不计算语言标签的loss
        source, future_target, past_target = self.dataset[index]
        source, target = self._make_source_target(source, future_target, past_target)
        source, target = self._maybe_add_bos(source, target)
        lang_index_src, lang_index_tgt = get_language_token_indexes(self.vocab, target)
        target[lang_index_src] = self.vocab.pad()
        target[lang_index_tgt] = self.vocab.pad()
        return {"id": index, "source": source, "target": target}
