from collections import OrderedDict

from fairseq import utils
from fairseq.models import register_model_architecture, register_model, FairseqMultiModel
from fairseq.models.multilingual_transformer import MultilingualTransformerModel, base_multilingual_architecture
from fairseq.models.transformer import transformer_wmt_en_de_big, TransformerEncoder, TransformerDecoder, TransformerConfig, Embedding
from fairseq.modules import transformer_layer
from fairseq.modules.transformer_layer import TransformerEncoderLayerBase, TransformerDecoderLayerBase
import torch.nn as nn
from fairseq.utils import safe_hasattr
from torch.nn import LayerNorm


@register_model("adapter_transformer")
class AdapterTransformer(MultilingualTransformerModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        pass

    def load_state_dict(self, state_dict, strict=True, model_cfg=None):
        return super(AdapterTransformer, self).load_state_dict(state_dict, strict=False, model_cfg=model_cfg)

    @classmethod
    def _get_module_class(cls, is_encoder, args, lang_dict, embed_tokens, langs):
        module_class = AdapterEncoder if is_encoder else AdapterDecoder
        return module_class(args, lang_dict, embed_tokens)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not safe_hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not safe_hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split("-")[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split("-")[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            args.share_encoder_embeddings = True
        if args.share_decoders:
            args.share_decoder_embeddings = True

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # build shared embeddings (if applicable)
        shared_encoder_embed_tokens, shared_decoder_embed_tokens = None, None
        if args.share_all_embeddings:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                dicts=task.dicts,
                langs=task.langs,
                embed_dim=args.encoder_embed_dim,
                build_embedding=build_embedding,
                pretrained_embed_path=args.encoder_embed_path,
            )
            shared_decoder_embed_tokens = shared_encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            if args.share_encoder_embeddings:
                shared_encoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=src_langs,
                    embed_dim=args.encoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.encoder_embed_path,
                )
            if args.share_decoder_embeddings:
                shared_decoder_embed_tokens = FairseqMultiModel.build_shared_embeddings(
                    dicts=task.dicts,
                    langs=tgt_langs,
                    embed_dim=args.decoder_embed_dim,
                    build_embedding=build_embedding,
                    pretrained_embed_path=args.decoder_embed_path,
                )

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}

        def get_encoder(lang):
            if lang not in lang_encoders:
                if shared_encoder_embed_tokens is not None:
                    encoder_embed_tokens = shared_encoder_embed_tokens
                else:
                    encoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.encoder_embed_dim,
                        args.encoder_embed_path,
                    )
            return cls._get_module_class(True, args, task.dicts[lang], encoder_embed_tokens, src_langs)

        def get_decoder(lang):
            if lang not in lang_decoders:
                if shared_decoder_embed_tokens is not None:
                    decoder_embed_tokens = shared_decoder_embed_tokens
                else:
                    decoder_embed_tokens = build_embedding(
                        task.dicts[lang],
                        args.decoder_embed_dim,
                        args.decoder_embed_path,
                    )

            return cls._get_module_class(False, args, task.dicts[lang], decoder_embed_tokens, tgt_langs)

        # shared encoders/decoders (if applicable)
        shared_encoder, shared_decoder = None, None
        if args.share_encoders:
            shared_encoder = get_encoder(src_langs[0])
        if args.share_decoders:
            shared_decoder = get_decoder(tgt_langs[0])

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = (
                shared_encoder if shared_encoder is not None else get_encoder(src)
            )
            decoders[lang_pair] = (
                shared_decoder if shared_decoder is not None else get_decoder(tgt)
            )

        model = AdapterTransformer(encoders, decoders)

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False
        return model


class AdapterEncoder(TransformerEncoder):
    def build_encoder_layer(self, args):
        return AdapterEncoderLayer(TransformerConfig.from_namespace(args), return_fc=self.return_fc)


class AdapterDecoder(TransformerDecoder):
    def build_decoder_layer(self, args, no_encoder_attn=False):
        return AdapterDecoderLayer(TransformerConfig.from_namespace(args), no_encoder_attn)


class AdapterEncoderLayer(TransformerEncoderLayerBase):
    def __init__(self, cfg, return_fc=False):
        super(AdapterEncoderLayer, self).__init__(cfg, return_fc)
        self.adapter = nn.Sequential(
                nn.Linear(cfg.decoder.embed_dim, cfg.decoder.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.decoder.embed_dim // 2, cfg.decoder.embed_dim),
                LayerNorm(cfg.decoder.embed_dim),
            )

    def forward(self, x, encoder_padding_mask, attn_mask=None):
        result = super(AdapterEncoderLayer, self).forward(x, encoder_padding_mask, attn_mask)
        result = self.adapter(result)
        return result


class AdapterDecoderLayer(TransformerDecoderLayerBase):
    def __init__(self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super(AdapterDecoderLayer, self).__init__(cfg, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.adapter = nn.Sequential(
                nn.Linear(cfg.decoder.embed_dim, cfg.decoder.embed_dim // 2),
                nn.ReLU(),
                nn.Linear(cfg.decoder.embed_dim // 2, cfg.decoder.embed_dim),
                LayerNorm(cfg.decoder.embed_dim),
            )

    def forward(self, *args, **kwargs):
        result = super(AdapterDecoderLayer, self).forward(*args, **kwargs)
        result = (self.adapter(result[0]), None, None,)
        return result


@register_model_architecture("adapter_transformer", "adapter_big")
def big_parameter_differentiation_architecture(args):
    transformer_wmt_en_de_big(args)

