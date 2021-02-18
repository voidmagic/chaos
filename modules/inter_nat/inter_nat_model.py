from fairseq.models.nat.levenshtein_transformer import LevenshteinTransformerModel, levenshtein_base_architecture
from fairseq.models import register_model, register_model_architecture


@register_model('nat_model')
class NatModelNoChange(LevenshteinTransformerModel):
    pass


@register_model_architecture('nat_model', 'nat_base')
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


@register_model_architecture('nat_model', 'nat_iwslt')
def levenshtein_transformer_iwslt(args):
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    levenshtein_base_architecture(args)
