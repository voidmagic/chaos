import torch.nn as nn
from fairseq import utils
from fairseq.models import register_model_architecture, register_model
from fairseq.models.multilingual_transformer import (
    base_multilingual_architecture,
    MultilingualTransformerModel
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, Linear
from fairseq.modules import LayerNorm


@register_model("adapter_share_transformer")
class AdapterSharedMultilingualTransformerModel(MultilingualTransformerModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)
        # self.make_share_components_modules()

    def make_share_components_modules(self):
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            # share encoder
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx].self_attn.k_proj = shared_model.encoder.layers[
                    layer_idx].self_attn.k_proj
                self.models[key].encoder.layers[layer_idx].self_attn.v_proj = shared_model.encoder.layers[
                    layer_idx].self_attn.v_proj
                self.models[key].encoder.layers[layer_idx].self_attn.q_proj = shared_model.encoder.layers[
                    layer_idx].self_attn.q_proj
                self.models[key].encoder.layers[layer_idx].self_attn.out_proj = shared_model.encoder.layers[
                    layer_idx].self_attn.out_proj
                self.models[key].encoder.layers[layer_idx].fc1 = shared_model.encoder.layers[layer_idx].fc1
                self.models[key].encoder.layers[layer_idx].fc2 = shared_model.encoder.layers[layer_idx].fc2
                self.models[key].encoder.layers[layer_idx].self_attn_layer_norm = shared_model.encoder.layers[
                    layer_idx].self_attn_layer_norm
                self.models[key].encoder.layers[layer_idx].final_layer_norm = shared_model.encoder.layers[
                    layer_idx].final_layer_norm

            # share decoder
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx].self_attn.k_proj = shared_model.decoder.layers[
                    layer_idx].self_attn.k_proj
                self.models[key].decoder.layers[layer_idx].self_attn.v_proj = shared_model.decoder.layers[
                    layer_idx].self_attn.v_proj
                self.models[key].decoder.layers[layer_idx].self_attn.q_proj = shared_model.decoder.layers[
                    layer_idx].self_attn.q_proj
                self.models[key].decoder.layers[layer_idx].self_attn.out_proj = shared_model.decoder.layers[
                    layer_idx].self_attn.out_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.k_proj = shared_model.decoder.layers[
                    layer_idx].encoder_attn.k_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.v_proj = shared_model.decoder.layers[
                    layer_idx].encoder_attn.v_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.q_proj = shared_model.decoder.layers[
                    layer_idx].encoder_attn.q_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.out_proj = shared_model.decoder.layers[
                    layer_idx].encoder_attn.out_proj
                self.models[key].decoder.layers[layer_idx].fc1 = shared_model.decoder.layers[layer_idx].fc1
                self.models[key].decoder.layers[layer_idx].fc2 = shared_model.decoder.layers[layer_idx].fc2
                self.models[key].decoder.layers[layer_idx].self_attn_layer_norm = shared_model.decoder.layers[
                    layer_idx].self_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].encoder_attn_layer_norm = shared_model.decoder.layers[
                    layer_idx].encoder_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].final_layer_norm = shared_model.decoder.layers[
                    layer_idx].final_layer_norm

    @classmethod
    def build_model(cls, args, task):
        super_model = super(AdapterSharedMultilingualTransformerModel, cls).build_model(args, task)
        encoders = {k: v.encoder for k, v in super_model.models.items()}
        decoders = {k: v.decoder for k, v in super_model.models.items()}
        return cls(encoders, decoders)

    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        module_class = AdapterEncoder if is_encoder else AdapterDecoder
        return module_class(args, lang_dict, embed_tokens)


class Adapter(nn.Module):
    def __init__(self, args, embed_tokens):
        super(Adapter, self).__init__()
        embed_dim = embed_tokens.embedding_dim
        self.layer_norm = LayerNorm(embed_dim)
        self.down_projection = Linear(embed_dim, args.encoder_ffn_embed_dim)
        self.activation_fn = utils.get_activation_fn(activation=getattr(args, "activation_fn", "relu"))
        self.up_projection = Linear(args.encoder_ffn_embed_dim, embed_dim)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.down_projection(x)
        x = self.activation_fn(x)
        x = self.up_projection(x)
        return residual + x


class AdapterEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(AdapterEncoder, self).__init__(args, dictionary, embed_tokens)
        self.adapter = Adapter(args, embed_tokens)

    def forward(self, *args, **kwargs):
        output = super(AdapterEncoder, self).forward(*args, **kwargs)
        return output._replace(encoder_out=self.adapter(output.encoder_out))


class AdapterDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super(AdapterDecoder, self).__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.adapter = Adapter(args, embed_tokens)

    def extract_features(self, *args, **kwargs):
        x, extra = super(AdapterDecoder, self).extract_features(*args, **kwargs)
        x = self.adapter(x)
        return x, extra


@register_model_architecture("adapter_share_transformer", "multilingual_adapt_base")
def multilingual_transformer_base(args):
    base_multilingual_architecture(args)
