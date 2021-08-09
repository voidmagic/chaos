from fairseq.models import register_model_architecture, register_model
from fairseq.models.multilingual_transformer import (
    base_multilingual_architecture,
    MultilingualTransformerModel,
    multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import transformer_wmt_en_de_big


@register_model("shared_multilingual_transformer")
class SharedMultilingualTransformerModel(MultilingualTransformerModel):

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        pass

    @classmethod
    def build_model(cls, args, task):
        assert args.share_encoders and args.share_decoders
        super_model = super(SharedMultilingualTransformerModel, cls).build_model(args, task)
        encoders = {k: v.encoder for k, v in super_model.models.items()}
        decoders = {k: v.decoder for k, v in super_model.models.items()}
        return cls(encoders, decoders)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.models[self.keys[0]].state_dict()

    def load_state_dict(self, state_dict, strict=True, args=None):
        return self.models[self.keys[0]].load_state_dict(state_dict, strict, args)


@register_model_architecture("shared_multilingual_transformer", "multilingual_shared_big")
def multilingual_transformer_big(args):
    transformer_wmt_en_de_big(args)
    base_multilingual_architecture(args)


@register_model_architecture("shared_multilingual_transformer", "multilingual_shared_base")
def multilingual_transformer_base(args):
    base_multilingual_architecture(args)


@register_model_architecture("shared_multilingual_transformer", "multilingual_shared_small")
def multilingual_transformer_small(args):
    multilingual_transformer_iwslt_de_en(args)


@register_model_architecture("multilingual_transformer", "multilingual_transformer_big")
def multilingual_transformer_small(args):
    multilingual_transformer_iwslt_de_en(args)

