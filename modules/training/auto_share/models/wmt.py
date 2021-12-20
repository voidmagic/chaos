from fairseq.models import register_model_architecture, register_model
from fairseq.models.multilingual_transformer import (
    base_multilingual_architecture,
    MultilingualTransformerModel,
    multilingual_transformer_iwslt_de_en
)
from fairseq.models.transformer import transformer_wmt_en_de_big


@register_model("wmt_share_transformer")
class WmtSharedMultilingualTransformerModel(MultilingualTransformerModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)
        self.make_share_components_modules()

    def make_share_components_modules(self):
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            # share encoder
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx].self_attn.k_proj = shared_model.encoder.layers[layer_idx].self_attn.k_proj
                # self.models[key].encoder.layers[layer_idx].self_attn.v_proj = shared_model.encoder.layers[layer_idx].self_attn.v_proj
                self.models[key].encoder.layers[layer_idx].self_attn.q_proj = shared_model.encoder.layers[layer_idx].self_attn.q_proj
                # self.models[key].encoder.layers[layer_idx].self_attn.out_proj = shared_model.encoder.layers[layer_idx].self_attn.out_proj
                # self.models[key].encoder.layers[layer_idx].fc1 = shared_model.encoder.layers[layer_idx].fc1
                # self.models[key].encoder.layers[layer_idx].fc2 = shared_model.encoder.layers[layer_idx].fc2
                self.models[key].encoder.layers[layer_idx].self_attn_layer_norm = shared_model.encoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].encoder.layers[layer_idx].final_layer_norm = shared_model.encoder.layers[layer_idx].final_layer_norm

            # share decoder
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx].self_attn.k_proj = shared_model.decoder.layers[layer_idx].self_attn.k_proj
                # self.models[key].decoder.layers[layer_idx].self_attn.v_proj = shared_model.decoder.layers[layer_idx].self_attn.v_proj
                self.models[key].decoder.layers[layer_idx].self_attn.q_proj = shared_model.decoder.layers[layer_idx].self_attn.q_proj
                # self.models[key].decoder.layers[layer_idx].self_attn.out_proj = shared_model.decoder.layers[layer_idx].self_attn.out_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.k_proj = shared_model.decoder.layers[layer_idx].encoder_attn.k_proj
                # self.models[key].decoder.layers[layer_idx].encoder_attn.v_proj = shared_model.decoder.layers[layer_idx].encoder_attn.v_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.q_proj = shared_model.decoder.layers[layer_idx].encoder_attn.q_proj
                # self.models[key].decoder.layers[layer_idx].encoder_attn.out_proj = shared_model.decoder.layers[layer_idx].encoder_attn.out_proj
                # self.models[key].decoder.layers[layer_idx].fc1 = shared_model.decoder.layers[layer_idx].fc1
                # self.models[key].decoder.layers[layer_idx].fc2 = shared_model.decoder.layers[layer_idx].fc2
                self.models[key].decoder.layers[layer_idx].self_attn_layer_norm = shared_model.decoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].encoder_attn_layer_norm = shared_model.decoder.layers[layer_idx].encoder_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].final_layer_norm = shared_model.decoder.layers[layer_idx].final_layer_norm

    @classmethod
    def build_model(cls, args, task):
        super_model = super(WmtSharedMultilingualTransformerModel, cls).build_model(args, task)
        encoders = {k: v.encoder for k, v in super_model.models.items()}
        decoders = {k: v.decoder for k, v in super_model.models.items()}
        return cls(encoders, decoders)


@register_model_architecture("wmt_share_transformer", "multilingual_wmt_big")
def multilingual_transformer_big(args):
    transformer_wmt_en_de_big(args)
    base_multilingual_architecture(args)


@register_model_architecture("wmt_share_transformer", "multilingual_wmt_base")
def multilingual_transformer_base(args):
    base_multilingual_architecture(args)


@register_model_architecture("wmt_share_transformer", "multilingual_wmt_small")
def multilingual_transformer_small(args):
    multilingual_transformer_iwslt_de_en(args)
