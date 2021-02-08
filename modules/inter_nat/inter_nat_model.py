from fairseq.models.nat.levenshtein_transformer import LevenshteinTransformerModel, levenshtein_base_architecture
from fairseq.models import register_model, register_model_architecture


@register_model('inter_nat')
class InterNatModel(LevenshteinTransformerModel):
    pass


@register_model_architecture('inter_nat', 'inter_nat_base')
def levenshtein_transformer_wmt_en_de(args):
    levenshtein_base_architecture(args)


@register_model_architecture('inter_nat', 'inter_nat_iwslt')
def levenshtein_transformer_iwslt(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    levenshtein_base_architecture(args)
