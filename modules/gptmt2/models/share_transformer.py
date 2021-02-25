from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerModel,
    base_architecture, transformer_wmt_en_de_big, transformer_iwslt_de_en)

from .decoder import Decoder
from .encoder import Encoder


@register_model("share_encoder_decoder_transformer")
class ShareEncoderDecoderTransformerModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        # --no-cross-attention：不包含cross attention，可单独
        # --share-parameters：共享除了cross attention之外的所有参数，可单独
        # --uni-dir-encoder：编码器使用单向，可单独
        # --layer-wise-attention：使用逐层注意力，可单独
        parser.add_argument('--share-parameters', default=False, action='store_true')
        parser.add_argument('--uni-dir-encoder', default=False, action='store_true')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true')

        # --language-embedding：语言编码
        # --layer-residual：跨层残差
        parser.add_argument('--language-embedding', default=False, action='store_true')
        parser.add_argument('--layer-residual', default=False, action='store_true')

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        if getattr(args, 'share_parameters', False):
            self.make_share_parameters()

    def make_share_parameters(self):
        # 共享除了language embedding以外的参数
        self.decoder.embed_positions = self.encoder.embed_positions
        for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
            dec_layer.self_attn.k_proj = enc_layer.self_attn.k_proj
            dec_layer.self_attn.v_proj = enc_layer.self_attn.v_proj
            dec_layer.self_attn.q_proj = enc_layer.self_attn.q_proj
            dec_layer.self_attn.out_proj = enc_layer.self_attn.out_proj
            dec_layer.self_attn_layer_norm = enc_layer.self_attn_layer_norm
            dec_layer.final_layer_norm = enc_layer.final_layer_norm
            dec_layer.fc1 = enc_layer.fc1
            dec_layer.fc2 = enc_layer.fc2

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        if getattr(args, 'uni_dir_encoder', False) or getattr(args, 'layer_wise_attention', False):
            return Encoder(args, src_dict, embed_tokens)
        else:
            return TransformerModel.build_encoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        if getattr(args, 'no_cross_attention', False) or getattr(args, 'layer_wise_attention', False):
            return Decoder(args, tgt_dict, embed_tokens)
        else:
            return TransformerModel.build_decoder(args, tgt_dict, embed_tokens)


@register_model_architecture("share_encoder_decoder_transformer", "share_small")
def share_encoder_decoder_transformer_base_arch(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    transformer_iwslt_de_en(args)


@register_model_architecture("share_encoder_decoder_transformer", "share_base")
def share_encoder_decoder_transformer_base_arch(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    base_architecture(args)


@register_model_architecture("share_encoder_decoder_transformer", "share_big")
def share_encoder_decoder_transformer_big_arch(args):
    args.encoder_normalize_before = True
    args.decoder_normalize_before = True
    transformer_wmt_en_de_big(args)
