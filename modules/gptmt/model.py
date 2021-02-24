from fairseq.models.transformer_lm import base_lm_architecture
from fairseq.models import register_model_architecture


@register_model_architecture("transformer_lm", "lm_iwslt")
def transformer_lm_iwslt(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 256)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 256)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "lm_iwslt_eqe")
def transformer_lm_iwslt_eq(args):
    args.decoder_layers = getattr(args, "decoder_layers", 4)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 256)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 256)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 384)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1536)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "lm_base")
def transformer_lm_base(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 512)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 512)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "lm_base_eqe")
def transformer_lm_base_eq(args):
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", 768)
    args.decoder_output_dim = getattr(args, "decoder_output_dim", 768)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 768)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 3072)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    base_lm_architecture(args)


