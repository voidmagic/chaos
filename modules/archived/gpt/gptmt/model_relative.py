from dataclasses import dataclass, field
from typing import List, Optional, Dict

import torch
from fairseq.models import register_model_architecture, register_model
from fairseq.models.transformer import TransformerDecoder
from fairseq.models.transformer_lm import base_lm_architecture, TransformerLanguageModel, TransformerLanguageModelConfig
from fairseq.modules import TransformerDecoderLayer
from torch import Tensor

from .modules import RelativePositionalEmbedding, RelativeMultiHeadAttention

DEFAULT_MAX_TARGET_POSITIONS = 1024


@dataclass
class LanguageModelRelativeConfig(TransformerLanguageModelConfig):
    share_rpe_across_heads: bool = field(default=False, metadata={"help": "share relative position across all heads"})


@register_model("lm_relative", dataclass=LanguageModelRelativeConfig)
class LanguageModelRelative(TransformerLanguageModel):
    @classmethod
    def build_model(cls, args, task):
        args.share_rpe_across_heads = getattr(args, 'share_rpe_across_heads', False)
        base_lm_architecture(args)
        args.no_token_positional_embeddings = True
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = getattr(args, "tokens_per_sample", DEFAULT_MAX_TARGET_POSITIONS)
        embed_tokens = cls.build_embedding(args, task.source_dictionary, args.decoder_input_dim)
        decoder = RelativeDecoder(args, task.target_dictionary, embed_tokens, no_encoder_attn=True)
        return cls(decoder)


class RelativeDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(RelativeDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if args.share_rpe_across_heads:
            self.positional_embedding_dim = self.embed_dim // self.args.decoder_attention_heads
        else:
            self.positional_embedding_dim = self.embed_dim
        self.layer_wise_position_embedding_k = RelativePositionalEmbedding(
            self.max_target_positions, self.positional_embedding_dim, self.padding_idx, vocab=self.dictionary)
        self.layer_wise_position_embedding_v = RelativePositionalEmbedding(
            self.max_target_positions, self.positional_embedding_dim, self.padding_idx, vocab=self.dictionary)


    def build_decoder_layer(self, args, no_encoder_attn=False):
        return RelativeDecoderLayer(args, no_encoder_attn)

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
    ):
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # 获取相对位置编码
        relative_position_embedding_k = self.layer_wise_position_embedding_k(prev_output_tokens)
        relative_position_embedding_v = self.layer_wise_position_embedding_v(prev_output_tokens)

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            relative_position_embedding_k = relative_position_embedding_k[:, -1:]
            relative_position_embedding_v = relative_position_embedding_v[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                relative_position_embedding=(relative_position_embedding_k, relative_position_embedding_v)
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

        return x, {"attn": [attn], "inner_states": inner_states}


class RelativeDecoderLayer(TransformerDecoderLayer):
    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        relative_position_embedding=None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
                    need_head_weights (bool, optional): return attention weights
                    for each head (default: return average over heads).
            relative_position_embedding: 相对位置编码
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
           """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            relative_position_embedding=relative_position_embedding
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
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return RelativeMultiHeadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


@register_model_architecture("lm_relative", "lm_relative_iwslt")
def transformer_lm_iwslt(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 256)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 256)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_lm_architecture(args)


@register_model_architecture("lm_relative", "lm_relative_iwslt_eqe")
def transformer_lm_iwslt_eq(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 256)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 256)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 384)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_lm_architecture(args)


@register_model_architecture("lm_relative", "lm_relative_base")
def transformer_lm_base(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 512)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 512)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    base_lm_architecture(args)


@register_model_architecture("lm_relative", "lm_relative_base_eqe")
def transformer_lm_base_eq(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 768)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    base_lm_architecture(args)
