
import torch
from fairseq import utils
from fairseq.modules import MultiheadAttention
import torch.nn.functional as F

from modules.sync_mnmt.task import Config


class InterAttention(MultiheadAttention):
    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, attn_mask=None, *args, **kwargs):
        num_lang = Config.n_lang

        tgt_len, bsz, embed_dim = query.size()
        bsz = bsz // num_lang

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(query)
        v = self.v_proj(query)

        q = torch.chunk(q, num_lang, dim=1)
        k = torch.chunk(k, num_lang, dim=1)
        v = torch.chunk(v, num_lang, dim=1)
        p = torch.chunk(key_padding_mask, num_lang, dim=0)

        attention_all_lang = []
        q_tensor = torch.cat(q, dim=1)
        for i in range(num_lang):
            k_tensor = torch.cat([k[(j + i) % len(k)] for j in range(len(k))], dim=1)
            v_tensor = torch.cat([v[(j + i) % len(v)] for j in range(len(v))], dim=1)
            p_tensor = torch.cat([p[(j + i) % len(p)] for j in range(len(p))], dim=0)

            # length x lang*batch x h
            attn = self.multi_head_attention(q_tensor, k_tensor, v_tensor, p_tensor, attn_mask)
            attention_all_lang.append(attn)

        if Config.manner == "gate":  # use gate for all languages
            lang_self_attention = attention_all_lang[0].view(-1, embed_dim).unsqueeze(1)
            lang_attention_q = self.cla_linear_q(lang_self_attention) * self.scaling  # lang query

            attention_all_lang = torch.stack(attention_all_lang, dim=0).view(num_lang, -1, embed_dim)
            attention_all_lang = attention_all_lang.transpose(0, 1)  # length*batch*lang x lang x h
            lang_attention_k = self.cla_linear_k(attention_all_lang)  # lang key
            lang_attention_v = self.cla_linear_v(attention_all_lang)  # lang value

            weight = torch.bmm(lang_attention_q, lang_attention_k.transpose(1, 2))
            weight = F.softmax(weight.float(), dim=-1).type_as(weight)
            attention_all_lang = torch.bmm(weight, lang_attention_v)

            attention_all_lang = attention_all_lang.view(tgt_len, num_lang * bsz, embed_dim)
        elif Config.manner == "tanh":
            attention_all_lang = torch.tanh(sum(attention_all_lang[1:])) * Config.tanh_weight + attention_all_lang[0]

        return attention_all_lang, None



    def multi_head_attention(self, q, k, v, key_padding_mask, attn_mask):
        query_len, batch_size, embed_dim = q.size()
        tgt_len = k.size(0)

        q = q.view(query_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(tgt_len, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.contiguous().view(-1, query_len, self.head_dim)  # (lang x batch x n_heads) x length x head_dim
        k = k.contiguous().view(-1, tgt_len, self.head_dim)  # (lang x batch x n_heads) x length x head_dim
        v = v.contiguous().view(-1, tgt_len, self.head_dim)  # (lang x batch x n_heads) x length x head_dim

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # -1 x tgt_len x tgt_len

        attn_mask = attn_mask.unsqueeze(0)
        attn_weights += attn_mask

        # don't attend to padding symbols
        attn_weights = attn_weights.view(batch_size, self.num_heads, query_len, tgt_len)
        attn_weights = attn_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_weights = attn_weights.view(-1, query_len, tgt_len)

        attn_prob = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)

        # attn: (batch x n_heads) x length x head_dim
        attn = torch.bmm(attn_prob, v).view(-1, query_len, self.head_dim)
        # attn: length x (batch x n_heads) x head_dim => length x batch x dim
        attn = attn.transpose(0, 1).contiguous().view(query_len, batch_size, embed_dim)
        return self.out_proj(attn)
