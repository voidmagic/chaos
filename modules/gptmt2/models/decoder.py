from typing import Optional, Dict, List

import torch
from fairseq.models.transformer import TransformerDecoder

from .decoder_layer import DecoderLayer
from .encoder import EncoderOut


class Decoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_cross_attention=True):
        super(Decoder, self).__init__(args, dictionary, embed_tokens, no_cross_attention)
        self.no_cross_attention = getattr(args, 'no_cross_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    def build_decoder_layer(self, args, no_encoder_attn=False):
        if getattr(args, 'no_cross_attention', False):
            return DecoderLayer(args, no_encoder_attn=True)
        else:
            return super(Decoder, self).build_decoder_layer(args, no_encoder_attn)

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
            if self.layer_wise_attention:
                encoder_state = encoder_out.encoder_states[idx]
            else:
                encoder_state = encoder_out.encoder_out

            x, layer_attn, _ = layer(
                x,
                encoder_state,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        x += torch.sum(encoder_out.encoder_out) * 0.  # 用到所有的结果，不然多卡会报错
        return x, {"attn": [attn], "inner_states": inner_states}

