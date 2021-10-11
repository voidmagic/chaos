from typing import Optional

from fairseq.models import register_model, register_model_architecture
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerModel, TransformerDecoder, base_architecture
from fairseq.modules import TransformerDecoderLayer

from modules.sync_mnmt.task import Config
from modules.sync_mnmt.utils.inter_attention import InterAttention


@register_model("sync_transformer")
class SyncMultilingualTransformer(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(args, tgt_dict, embed_tokens)


class Decoder(TransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return DecoderLayer(args, no_encoder_attn)

    def forward(self, prev_output_tokens, encoder_out: Optional[EncoderOut] = None, *args, **kwargs):
        n_lang = Config.n_lang
        encoder_out = EncoderOut(
            encoder_out=encoder_out.encoder_out.repeat(1, n_lang, 1),
            encoder_embedding=encoder_out.encoder_embedding.repeat(n_lang, 1, 1),
            encoder_padding_mask=encoder_out.encoder_padding_mask.repeat(n_lang, 1),
            encoder_states=[state.repeat(1, n_lang, 1) for state in encoder_out.encoder_states],
            src_tokens=None,
            src_lengths=None,
        )
        return super(Decoder, self).forward(prev_output_tokens, encoder_out, *args, **kwargs)


class DecoderLayer(TransformerDecoderLayer):
    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return InterAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )




@register_model_architecture("sync_transformer", "sync_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

