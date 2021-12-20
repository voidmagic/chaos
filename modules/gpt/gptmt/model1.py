from fairseq.models import register_model_architecture
from fairseq.model_parallel.models.transformer_lm import base_lm_architecture


@register_model_architecture("model_parallel_transformer_lm", "zhgpt_2")
def zh_gpt_2(args):
    hidden_dim = 2048
    num_header = 32
    num_layers = 40
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", hidden_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", hidden_dim * 4)
    args.decoder_layers = getattr(args, "decoder_layers", num_layers)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", num_header)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)


@register_model_architecture("transformer_lm", "zhgpt_2_infer")
def zh_gpt_2(args):
    hidden_dim = 2048
    num_header = 32
    num_layers = 40
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", hidden_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", hidden_dim * 4)
    args.decoder_layers = getattr(args, "decoder_layers", num_layers)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", num_header)
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    base_lm_architecture(args)

