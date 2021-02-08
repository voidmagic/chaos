from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
from fairseq.models.transformer import TransformerModel, base_architecture, transformer_wmt_en_de_big
from fairseq.models import register_model_architecture, register_model


@register_model('pretrain_transformer')
class PretrainMachineTranslationModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = F.relu
        self.dense = nn.Linear(args.decoder_embed_dim, args.decoder_embed_dim)
        self.classifier = nn.Linear(args.decoder_embed_dim, 2)
        self.register_buffer('arange', torch.arange(sum(self.max_positions()) * 10))

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        return_classification: bool = False,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        if return_classification:
            last_token_index = (prev_output_tokens != self.decoder.padding_idx).sum(1) - 1
            last_token_index += self.arange[:prev_output_tokens.size(0)] * prev_output_tokens.size(1)
            decoder_state = decoder_out[1]['inner_states'][-1].transpose(0, 1).contiguous()
            decoder_state = decoder_state.view(-1, decoder_state.size(-1))
            last_token_state = decoder_state.index_select(0, last_token_index)
            last_token_state = self.dropout(last_token_state)
            last_token_state = self.dense(last_token_state)
            last_token_state = self.activation(last_token_state)
            last_token_state = self.classifier(last_token_state)
            return decoder_out, last_token_state
        return decoder_out


@register_model_architecture("pretrain_transformer", "ptmt_base")
def pretrain_transformer_wmt_en_de(args):
    base_architecture(args)


@register_model_architecture("pretrain_transformer", "ptmt_big")
def pretrain_transformer_wmt_en_de(args):
    transformer_wmt_en_de_big(args)
