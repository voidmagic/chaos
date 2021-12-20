from typing import Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerEncoder

from .encoder_layer import EncoderLayer
from .layer_drop import LayerDropModuleList


class Encoder(TransformerEncoder):
    def __init__(self, args, src_dict, embed_tokens):
        super(Encoder, self).__init__(args, src_dict, embed_tokens)

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [self.build_encoder_layer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)

        self._future_mask = torch.empty(0)
        self.uni_dir = getattr(args, 'uni_dir_encoder', False)
        if getattr(args, 'language_embedding', False):
            self.language_embedding = nn.Parameter(torch.Tensor(2, embed_tokens.embedding_dim))
            nn.init.normal_(self.language_embedding, mean=0, std=embed_tokens.embedding_dim ** -0.5)
        else:
            self.language_embedding = None
        self.layer_residual = getattr(args, 'layer_residual', False)

    def build_encoder_layer(self, args):
        return EncoderLayer(args)

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = True,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        self_attn_mask = self.buffered_future_mask(x) if self.uni_dir else None
        encoder_states = [x]
        # encoder layers

        layers = self.layers.with_drop(reset=True) if isinstance(self.layers, LayerDropModuleList) else self.layers
        layers = [layer for layer in layers]
        x += sum([empty_layer for empty_layer in layers if isinstance(empty_layer, torch.Tensor)] + [0.]) * 0.
        layers = [layer for layer in layers if not isinstance(layer, torch.Tensor)]
        for layer in layers:
            y = x
            x = layer(x, encoder_padding_mask, self_attn_mask, language_embedding=self.language_embedding)
            if self.layer_residual:
                x = x + y
            encoder_states.append(x)   # 把embedding的保存下来，最后一层不要了

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None,
        )

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
                self._future_mask.size(0) == 0
                or (not self._future_mask.device == tensor.device)
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]
