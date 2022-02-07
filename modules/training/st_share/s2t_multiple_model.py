from fairseq.models import FairseqMultiModel, register_model, register_model_architecture
from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerModel, base_architecture


@register_model("speech_transformer")
class S2TMultiModel(FairseqMultiModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @classmethod
    def add_args(cls, parser):
        return S2TTransformerModel.add_args(parser)

    @classmethod
    def build_model(cls, args, task):
        model_keys = args.model_keys

        model = S2TTransformerModel.build_model(args, task)

        encoders = {
            key: model.encoder
            for key in model_keys.split(",")
        }

        decoders = {
            key: model.decoder
            for key in model_keys.split(",")
        }
        return cls(encoders, decoders)

    def max_positions(self):
        max_positions = super(S2TMultiModel, self).max_positions()
        return list(max_positions.values())[0]

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        state_dict_full = {}
        if not any([k.startswith("model") for k, _ in state_dict.items()]):
            # load state dict from single shared model
            for k, value in state_dict.items():
                for model_key in self.keys:
                    state_dict_full["models.{}.{}".format(model_key, k)] = value
        else:
            state_dict_full = state_dict

        state_dict_subset = state_dict_full.copy()
        for k, _ in state_dict_full.items():
            assert k.startswith("models.")
            lang_pair = k.split(".")[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, model_cfg=model_cfg)


@register_model_architecture("speech_transformer", "speech_transformer")
def s2t_transformer_m(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512 * 4)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.15)
    base_architecture(args)
