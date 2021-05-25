from typing import Optional

import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.modules.transformer_layer import TransformerEncoderLayer
from fairseq.models.transformer import TransformerModel, TransformerEncoder, transformer_iwslt_de_en
from torch import Tensor


@register_model("temporal_transformer")
class TemporalTransformer(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TemporalEncoder(args, src_dict, embed_tokens)


class TemporalEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(TemporalEncoder, self).__init__(args, dictionary, embed_tokens)
        self.embed_positions = None

    def build_encoder_layer(self, args):
        return TemporalEncoderLayer(args)


class TemporalEncoderLayer(TransformerEncoderLayer):
    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


@register_model_architecture("temporal_transformer", "temporal_transformer_iwslt_de_en")
def temporal_transformer_iwslt_de_en(args):
    transformer_iwslt_de_en(args)
