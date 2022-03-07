import torch
from fairseq.models import FairseqMultiModel, register_model, register_model_architecture
from fairseq.models.speech_to_text.s2t_transformer import S2TTransformerModel, base_architecture
import os
import torch.nn as nn
import random
from sklearn.metrics.pairwise import cosine_similarity


def name2module(module, name):
    def _generator(_module):
        for part in name.split('.'):
            _module = getattr(_module, part)
            yield _module
    return [module] + list(_generator(module))


def build_model_shared(modules, share_keys, encoder_nps=0, decoder_nps=0):
    if share_keys == "universal":
        modules = {key: list(modules.values())[0] for key in modules.keys()}
    elif share_keys == "individual":
        pass
    elif share_keys == "partial":
        # share K, V
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].encoder_attn.k_proj = first_module.layers[layer].encoder_attn.k_proj
                module.layers[layer].encoder_attn.v_proj = first_module.layers[layer].encoder_attn.v_proj
                module.layers[layer].self_attn.k_proj = first_module.layers[layer].self_attn.k_proj
                module.layers[layer].self_attn.v_proj = first_module.layers[layer].self_attn.v_proj
    elif share_keys == "deeper":
        # share deeper layers, keep shallow layers language-specific
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(3, 6):
                module.layers[layer] = first_module.layers[layer]
    elif share_keys == "shallower":
        # share deeper layers, keep shallow layers language-specific
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(3):
                module.layers[layer] = first_module.layers[layer]
    elif share_keys == "partial1":
        # share: K, Q, V
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].encoder_attn.k_proj = first_module.layers[layer].encoder_attn.k_proj
                module.layers[layer].encoder_attn.v_proj = first_module.layers[layer].encoder_attn.v_proj
                module.layers[layer].encoder_attn.q_proj = first_module.layers[layer].encoder_attn.q_proj
                module.layers[layer].self_attn.k_proj = first_module.layers[layer].self_attn.k_proj
                module.layers[layer].self_attn.v_proj = first_module.layers[layer].self_attn.v_proj
                module.layers[layer].self_attn.q_proj = first_module.layers[layer].self_attn.q_proj
    elif share_keys == "partial2":
        # share: k1, q1, v1, f1
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].self_attn.k_proj = first_module.layers[layer].self_attn.k_proj
                module.layers[layer].self_attn.v_proj = first_module.layers[layer].self_attn.v_proj
                module.layers[layer].self_attn.q_proj = first_module.layers[layer].self_attn.q_proj
                module.layers[layer].self_attn.out_proj = first_module.layers[layer].self_attn.out_proj
    elif share_keys == "partial3":
        # share: k2, q2, v2, f2
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].encoder_attn.k_proj = first_module.layers[layer].encoder_attn.k_proj
                module.layers[layer].encoder_attn.v_proj = first_module.layers[layer].encoder_attn.v_proj
                module.layers[layer].encoder_attn.q_proj = first_module.layers[layer].encoder_attn.q_proj
                module.layers[layer].encoder_attn.out_proj = first_module.layers[layer].encoder_attn.out_proj
    elif share_keys == "partial4":
        # share: k2, q2, v2, f2
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].encoder_attn.k_proj = first_module.layers[layer].encoder_attn.k_proj
                module.layers[layer].encoder_attn.q_proj = first_module.layers[layer].encoder_attn.q_proj
                module.layers[layer].self_attn.k_proj = first_module.layers[layer].self_attn.k_proj
                module.layers[layer].self_attn.q_proj = first_module.layers[layer].self_attn.q_proj
    elif share_keys == "partial5":
        # share: K, Q, V
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].encoder_attn.k_proj = first_module.layers[layer].encoder_attn.k_proj
                module.layers[layer].encoder_attn.v_proj = first_module.layers[layer].encoder_attn.v_proj
                module.layers[layer].encoder_attn.q_proj = first_module.layers[layer].encoder_attn.q_proj
                module.layers[layer].encoder_attn.out_proj = first_module.layers[layer].encoder_attn.out_proj
                module.layers[layer].self_attn.k_proj = first_module.layers[layer].self_attn.k_proj
                module.layers[layer].self_attn.v_proj = first_module.layers[layer].self_attn.v_proj
                module.layers[layer].self_attn.q_proj = first_module.layers[layer].self_attn.q_proj
                module.layers[layer].self_attn.out_proj = first_module.layers[layer].self_attn.out_proj
    elif share_keys == "partial6":
        # share: L1 L2
        first_module = list(modules.values())[0]
        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
            for layer in range(6):
                module.layers[layer].fc1 = first_module.layers[layer].fc1
                module.layers[layer].fc2 = first_module.layers[layer].fc2
    elif share_keys.startswith("random"):
        # all
        random.seed(0)
        modules = nn.ModuleDict(modules)
        ratio = float(share_keys.strip("random"))
        total_param_threshold = (encoder_nps + decoder_nps) * ratio
        decoders_threshold = total_param_threshold - encoder_nps
        first_module = list(modules.values())[0]

        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection

        while sum(p.numel() for p in modules.parameters()) > decoders_threshold:
            # random select parameter for share
            # select 2 keys
            languages = random.sample(list(modules.keys()), 2)
            keys = random.choice([n.rstrip('.bias').rstrip('.weight') for n, _ in first_module.named_parameters() if n.startswith('layers') and 'norm' not in n])
            setattr(name2module(modules[languages[1]], keys)[-2], keys.split(".")[-1], name2module(modules[languages[0]], keys)[-1])
    elif share_keys.startswith("pdiff"):
        # all
        random.seed(0)
        modules = nn.ModuleDict(modules)
        ratio = float(share_keys.strip("pdiff"))
        total_param_threshold = (encoder_nps + decoder_nps) * ratio
        decoders_threshold = total_param_threshold - encoder_nps
        first_module = list(modules.values())[0]

        for key, module in modules.items():
            module.embed_tokens = first_module.embed_tokens
            module.output_projection = first_module.output_projection
        # load pre_computed gradients
        gradients = {
            key: torch.load("{}/{}.pt".format(os.environ['ST_GRADIENT_PATH'], key))
            for key in modules.keys()
        }

        # get gradient sim rank
        rank = []
        for lang1 in gradients.keys():
            for lang2 in gradients.keys():
                if lang1 == lang2:
                    continue
                for p_name in ["decoder." + n for n, _ in first_module.named_parameters() if n.startswith('layers') and 'norm' not in n]:
                    grad_1 = gradients[lang1][p_name]
                    grad_2 = gradients[lang2][p_name]
                    sim = cosine_similarity(grad_1.view(1, -1), grad_2.view(1, -1))[0][0]
                    rank.append((p_name.lstrip("decoder."), lang1, lang2, float(sim)))

        # rank high -> low, similar first. similar should be shared
        rank = sorted(rank, key=lambda quad: quad[-1], reverse=True)
        for p_name, lang1, lang2, _ in rank:
            setattr(name2module(modules[lang1], p_name)[-2], p_name.split(".")[-1], name2module(modules[lang2], p_name)[-1])
            if sum(p.numel() for p in modules.parameters()) < decoders_threshold:
                break
    else:
        raise NotImplementedError(share_keys)
    return modules


@register_model("speech_transformer")
class S2TMultiModel(FairseqMultiModel):
    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @classmethod
    def build_model(cls, args, task):
        model_keys = os.environ['ST_MODEL_KEY']
        model = {key: S2TTransformerModel.build_model(args, task) for key in model_keys.split(",")}
        encoders = {key: model[key].encoder for key in model_keys.split(",")}
        decoders = {key: model[key].decoder for key in model_keys.split(",")}
        encoder_nps = sum(p.numel() for p in encoders[model_keys.split(",")[0]].parameters())
        decoder_nps = sum(p.numel() for p in decoders[model_keys.split(",")[0]].parameters())

        encoders = build_model_shared(encoders, "universal")
        decoders = build_model_shared(decoders, os.environ['ST_SHARE_KEY'], encoder_nps=encoder_nps, decoder_nps=decoder_nps)
        model = cls(encoders, decoders)
        return model

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
    args.dropout = getattr(args, "activation_dropout", 0.15)
    args.dropout = getattr(args, "attention_dropout", 0.15)
    base_architecture(args)
