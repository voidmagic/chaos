from fairseq.models import register_model, register_model_architecture
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import base_architecture


@register_model('parameter_differentiation_model')
class ParameterDifferentiationModel(MultilingualTransformerModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx] = shared_model.encoder.layers[layer_idx]
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx] = shared_model.decoder.layers[layer_idx]

    @classmethod
    def build_model(cls, args, task):
        model = super(ParameterDifferentiationModel, cls).build_model(args, task)
        encoders = {key: model.models[key].encoder for key in model.keys}
        decoders = {key: model.models[key].decoder for key in model.keys}
        return cls(encoders, decoders)


@register_model_architecture("parameter_differentiation_model", "parameter_differentiation_base_model")
def base_parameter_differentiation_architecture(args):
    base_architecture(args)
