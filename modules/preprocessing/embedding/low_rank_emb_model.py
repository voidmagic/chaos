import torch
import torch.nn as nn
from fairseq.models import register_model_architecture, register_model
from fairseq.models.transformer import TransformerModel, transformer_iwslt_de_en, TransformerConfig, TransformerEncoderBase


@register_model("emb_transformer")
class EmbedTransformer(TransformerModel):
    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return EmbedEncoder(TransformerConfig.from_namespace(args), src_dict, embed_tokens)


class EmbedEncoder(TransformerEncoderBase):
    def __init__(self, cfg, dictionary, embed_tokens, return_fc=False):
        super(EmbedEncoder, self).__init__(cfg, dictionary, embed_tokens, return_fc)
        num_langs = len(cfg.langs)
        self.offset = len(dictionary.symbols) - num_langs
        self.weight_l = nn.Parameter(torch.zeros((embed_tokens.embedding_dim, 10)))
        self.weight_s = nn.Parameter(torch.zeros((10, num_langs)))
        nn.init.normal_(self.weight_l, mean=0, std=embed_tokens.embedding_dim ** -0.5)
        nn.init.normal_(self.weight_s, mean=0, std=embed_tokens.embedding_dim ** -0.5)

    def forward_embedding(self, src_tokens, token_embedding=None):
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)

        task_embedding = torch.matmul(self.weight_l, self.weight_s)
        task_embedding = torch.index_select(task_embedding, dim=1, index=(src_tokens[:, 0] - self.offset))
        token_embedding[:, 0, :] = task_embedding.T

        # token_embedding
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed


@register_model_architecture("emb_transformer", "emb_iwslt")
def transformer_iwslt_de_en_emb(args):
    transformer_iwslt_de_en(args)

