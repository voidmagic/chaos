import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerDecoder, base_architecture
from fairseq.modules import TransformerDecoderLayer

from modules.sync_mnmt.task import Config
from modules.sync_mnmt.utils.inter_attention import InterAttention


@register_model("sync_transformer")
class SyncMultilingualTransformer(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, *args, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths)
        new_order = torch.arange(src_tokens.size(0)).repeat(Config.n_lang).to(src_tokens.device).long()
        encoder_out = self.encoder.reorder_encoder_out(encoder_out, new_order)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out)
        return decoder_out


class Decoder(TransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return DecoderLayer(args)


class DecoderLayer(TransformerDecoderLayer):
    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return InterAttention(embed_dim, args.decoder_attention_heads, dropout=args.attention_dropout, self_attention=True)


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


@register_model_architecture("sync_transformer", "sync_base")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
