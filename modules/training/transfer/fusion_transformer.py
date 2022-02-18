import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en, \
    TransformerEncoderBase
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
)


@register_model("fusion_transformer")
class FusionTransformerModel(TransformerModel):
    @classmethod
    def add_args(cls, parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--model-1-path', type=str, default=None)
        parser.add_argument('--model-2-path', type=str, default=None)

    @classmethod
    def build_model(cls, args, task):
        # model_1: main model
        model_1 = super(FusionTransformerModel, cls).build_model(args, task)
        if args.model_1_path is not None:
            model_1.load_state_dict(torch.load(args.model_1_path)["model"])

        # model_2: aux model
        model_2 = super(FusionTransformerModel, cls).build_model(args, task)
        if args.model_2_path is not None:
            model_2.load_state_dict(torch.load(args.model_2_path)["model"])

        overall_model = model_1
        overall_model.encoder.aux_encoder = model_2.encoder
        overall_model.encoder.gate_linear_1 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=True)
        overall_model.encoder.gate_linear_2 = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim, bias=True)

        for name, p in overall_model.named_parameters():
            if 'encoder' in name and 'gate_linear' not in name:
                p.requires_grad = False

        return overall_model

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return FusionEncoder(TransformerConfig.from_namespace(args), src_dict, embed_tokens)


class FusionEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens):
        super(FusionEncoder, self).__init__(cfg, dictionary, embed_tokens)
        self.aux_encoder = None
        self.gate_linear_1 = None
        self.gate_linear_2 = None

    def forward(self, src_tokens, src_lengths=None, return_all_hiddens=False, token_embeddings=None):
        encoder_out = self.forward_scriptable(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        encoder_out = self.fusion_aux(encoder_out, src_tokens, src_lengths)
        return encoder_out

    def fusion_aux(self, encoder_out, src_tokens, src_lengths):
        if self.aux_encoder is None:
            return encoder_out
        encoder_out_aux = self.aux_encoder(src_tokens, src_lengths=src_lengths)
        first_states = encoder_out['encoder_out'][0]
        second_states = encoder_out_aux['encoder_out'][0]

        gate = F.sigmoid(self.gate_linear_1(first_states) + self.gate_linear_2(second_states))
        final_states = first_states + gate * second_states
        encoder_out['encoder_out'][0] = final_states
        return encoder_out


@register_model_architecture("fusion_transformer", "fusion_transformer_iwslt")
def fusion_transformer_iwslt(args):
    transformer_iwslt_de_en(args)

