from fairseq.models.transformer import TransformerModel, TransformerDecoder


class SyncMultilingualTransformer(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


