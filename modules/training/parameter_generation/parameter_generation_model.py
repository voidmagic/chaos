import torch
import torch.nn as nn
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, base_architecture, transformer_iwslt_de_en


@register_model("parameter_generation_transformer")
class ParameterGenerationModel(TransformerModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        # disable parameter training
        for parameter in self.parameters():
            parameter.requires_grad = False

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        return super(ParameterGenerationModel, self).load_state_dict(state_dict, False, model_cfg, args)


@register_model_architecture("parameter_generation_transformer", "parameter_generation_transformer")
def parameter_generation_transformer_base(args):
    base_architecture(args)


@register_model_architecture("parameter_generation_transformer", "parameter_generation_transformer_iwslt")
def parameter_generation_transformer_iwslt(args):
    transformer_iwslt_de_en(args)

