# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..utils.dataset import get_language_token_indexes


def get_absolute_position(tensor, position_offset, vocab):
    positions = torch.ones_like(tensor).long()
    for m in range(tensor.size(0)):
        # 对第一个句子，找出两个语言标签的位置
        src_lang_index, tgt_lang_index = get_language_token_indexes(vocab, tensor[m])
        if src_lang_index is not None and tgt_lang_index is not None:
            positions[m, :tgt_lang_index] = torch.arange(1, int(tgt_lang_index) + 1) + position_offset
            positions[m, tgt_lang_index:] = torch.arange(1, tensor.size(1) - int(tgt_lang_index) + 1)
        else:
            positions[m] = torch.arange(1, positions.size(1) + 1) + position_offset
    return positions


def make_positions(tensor, padding_idx: int, onnx_trace: bool = False):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx


class RelativePositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, max_position: int, embedding_dim: int, padding_idx: int, vocab=None):
        super().__init__(max_position * 4 + 5, embedding_dim, padding_idx)
        self.max_position = max_position  # e.g. 512
        self.vocab = vocab
        self.onnx_trace = False

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.weight[self.padding_idx], 0)

    def forward(
        self,
        input: Tensor,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        positions: Optional[Tensor] = None,
    ):
        # 获取序列中每个位置相对其他位置的位置编码

        positions = get_absolute_position(input, self.max_position, self.vocab)
        positions_relative = positions.unsqueeze(1)
        positions_relative = positions_relative - positions_relative.transpose(1, 2)
        positions_relative += 2 * self.max_position + self.padding_idx
        embedding = F.embedding(
            positions_relative,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return embedding
