
from collections import OrderedDict

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.multilingual_transformer import MultilingualTransformerModel
from fairseq.models.transformer import (
    Embedding,
    base_architecture, transformer_wmt_en_de_big,
)


@register_model('auto_share_multilingual')
class Model(MultilingualTransformerModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        pass

    def __init__(self, encoders, decoders, share=False, granularity=None):
        super().__init__(encoders, decoders)
        self.make_share_components(share, granularity)

    def make_share_components(self, share, granularity):
        # 在训练的时候，share=True，然后自动学习那些是不需要share的；在测试的时候，share=False，每个语言对独享一组参数！
        if not share:
            return
        if granularity == 'parameter':
            self.make_share_components_params()
        elif granularity == 'module':
            self.make_share_components_modules()
        elif granularity == 'layer':
            self.make_share_components_layers()
        else:
            raise NotImplementedError(granularity)

    def make_share_components_params(self):
        shared_model = self.models[self.keys[0]]
        for name, parameter in shared_model.named_parameters():
            if 'layers' not in name:
                continue

            for key in self.keys[1:]:
                module = self.models[key]
                for module_name in name.split('.')[:-1]:
                    module = getattr(module, module_name)
                setattr(module, name.split('.')[-1], parameter)

    def make_share_components_modules(self):
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            # share encoder
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx].self_attn.k_proj = shared_model.encoder.layers[layer_idx].self_attn.k_proj
                self.models[key].encoder.layers[layer_idx].self_attn.v_proj = shared_model.encoder.layers[layer_idx].self_attn.v_proj
                self.models[key].encoder.layers[layer_idx].self_attn.q_proj = shared_model.encoder.layers[layer_idx].self_attn.q_proj
                self.models[key].encoder.layers[layer_idx].self_attn.out_proj = shared_model.encoder.layers[layer_idx].self_attn.out_proj
                self.models[key].encoder.layers[layer_idx].fc1 = shared_model.encoder.layers[layer_idx].fc1
                self.models[key].encoder.layers[layer_idx].fc2 = shared_model.encoder.layers[layer_idx].fc2
                self.models[key].encoder.layers[layer_idx].self_attn_layer_norm = shared_model.encoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].encoder.layers[layer_idx].final_layer_norm = shared_model.encoder.layers[layer_idx].final_layer_norm

            # share decoder
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx].self_attn.k_proj = shared_model.decoder.layers[layer_idx].self_attn.k_proj
                self.models[key].decoder.layers[layer_idx].self_attn.v_proj = shared_model.decoder.layers[layer_idx].self_attn.v_proj
                self.models[key].decoder.layers[layer_idx].self_attn.q_proj = shared_model.decoder.layers[layer_idx].self_attn.q_proj
                self.models[key].decoder.layers[layer_idx].self_attn.out_proj = shared_model.decoder.layers[layer_idx].self_attn.out_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.k_proj = shared_model.decoder.layers[layer_idx].encoder_attn.k_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.v_proj = shared_model.decoder.layers[layer_idx].encoder_attn.v_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.q_proj = shared_model.decoder.layers[layer_idx].encoder_attn.q_proj
                self.models[key].decoder.layers[layer_idx].encoder_attn.out_proj = shared_model.decoder.layers[layer_idx].encoder_attn.out_proj
                self.models[key].decoder.layers[layer_idx].fc1 = shared_model.decoder.layers[layer_idx].fc1
                self.models[key].decoder.layers[layer_idx].fc2 = shared_model.decoder.layers[layer_idx].fc2
                self.models[key].decoder.layers[layer_idx].self_attn_layer_norm = shared_model.decoder.layers[layer_idx].self_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].encoder_attn_layer_norm = shared_model.decoder.layers[layer_idx].encoder_attn_layer_norm
                self.models[key].decoder.layers[layer_idx].final_layer_norm = shared_model.decoder.layers[layer_idx].final_layer_norm

    def make_share_components_layers(self):
        shared_model = self.models[self.keys[0]]
        for key in self.keys[1:]:
            for layer_idx in range(len(shared_model.encoder.layers)):
                self.models[key].encoder.layers[layer_idx] = shared_model.encoder.layers[layer_idx]
            for layer_idx in range(len(shared_model.decoder.layers)):
                self.models[key].decoder.layers[layer_idx] = shared_model.decoder.layers[layer_idx]


    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_multilingual_auto_share_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 1024
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 1024

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        if args.encoder_embed_dim != args.decoder_embed_dim:
            raise ValueError("--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim")

        shared_dict = next(iter(task.dicts.values()))
        shared_embed_tokens = build_embedding(shared_dict, args.encoder_embed_dim)
        args.share_decoder_input_output_embed = True

        def get_encoder():
            return cls._get_module_class(True, args, shared_dict, shared_embed_tokens, src_langs)

        def get_decoder():
            return cls._get_module_class(False, args, shared_dict, shared_embed_tokens, tgt_langs)

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = get_encoder()
            decoders[lang_pair] = get_decoder()
        return cls(encoders, decoders, share=task.training, granularity=getattr(args, 'split_granularity', 'layer'))


@register_model_architecture("auto_share_multilingual", "auto_base")
def base_multilingual_auto_share_architecture(args):
    base_architecture(args)


@register_model_architecture("auto_share_multilingual", "auto_small")
def multilingual_transformer_auto_share_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_multilingual_auto_share_architecture(args)


@register_model_architecture("auto_share_multilingual", "auto_big")
def big_multilingual_auto_share_architecture(args):
    transformer_wmt_en_de_big(args)
