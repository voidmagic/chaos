from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en
from fairseq.models import register_model, register_model_architecture


@register_model("no_pe_model")
class PositionalEmbeddingEmptyDecoderModel(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        args.no_token_positional_embeddings = True
        return TransformerModel.build_decoder(args, tgt_dict, embed_tokens)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        args.no_token_positional_embeddings = False
        return TransformerModel.build_encoder(args, src_dict, embed_tokens)


@register_model_architecture("no_pe_model", "iwslt_no_pe")
def arch_iwslt_no_pe(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    transformer_iwslt_de_en(args)
