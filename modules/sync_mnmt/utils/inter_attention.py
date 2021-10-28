import math

import torch
import torch.nn as nn
from fairseq.modules import MultiheadAttention
import torch.nn.functional as F

from modules.sync_mnmt.task import Config


class InterAttention(MultiheadAttention):
    def __init__(self, embed_dim, *args, **kwargs):
        super(InterAttention, self).__init__(embed_dim, *args, **kwargs)
        self.cla_linear_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cla_linear_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.cla_linear_v = nn.Linear(embed_dim, embed_dim, bias=True)

        nn.init.xavier_uniform_(self.cla_linear_q.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.cla_linear_k.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.cla_linear_v.weight, gain=1 / math.sqrt(2))

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, attn_mask=None, *args, **kwargs):
        num_lang = Config.n_lang
        if key_padding_mask is None:
            key_padding_mask = torch.zeros_like(key[:, :, 0]).bool().transpose(0, 1)

        tgt_len, bsz, embed_dim = query.size()
        bsz = bsz // num_lang

        q = self.q_proj(query) * self.scaling
        k = self.k_proj(query)
        v = self.v_proj(query)

        if incremental_state is not None:
            # 如果有之前的state，拼接
            saved_state = incremental_state.get(self, {})
            if 'self_prev_key' in saved_state:
                prev_key = saved_state['self_prev_key']
                k = torch.cat((prev_key, k), dim=0)
            if 'self_prev_value' in saved_state:
                prev_value = saved_state['self_prev_value']
                v = torch.cat((prev_value, v), dim=0)
            if 'prev_key_padding_mask' in saved_state:
                prev_key_padding_mask = saved_state['prev_key_padding_mask']
                key_padding_mask = torch.cat((prev_key_padding_mask, key_padding_mask), dim=1)
            saved_state['self_prev_key'] = k
            saved_state['self_prev_value'] = v
            saved_state['prev_key_padding_mask'] = key_padding_mask
            incremental_state[self] = saved_state

        k = list(torch.chunk(k, num_lang, dim=1))
        v = list(torch.chunk(v, num_lang, dim=1))
        p = list(torch.chunk(key_padding_mask, num_lang, dim=0))

        attention_list = []
        for i in range(num_lang):  # 拼接后
            k_i = torch.cat([k[(j + i) % len(k)] for j in range(len(k))], dim=1)
            v_i = torch.cat([v[(j + i) % len(v)] for j in range(len(v))], dim=1)
            p_i = torch.cat([p[(j + i) % len(p)] for j in range(len(p))], dim=0)
            attention_list.append(self.multi_head_attention(q, k_i, v_i, p_i, attn_mask))  # length x lang*batch x h, lang表示第k个语言到第k+i个语言的注意力

        if Config.manner == "none":
            output = attention_list[0]
        elif Config.manner == "gate":  # use gate for all languages
            cla_q = self.cla_linear_q(attention_list[0].view(-1, embed_dim).unsqueeze(1)) * self.scaling  # length*lang*batch x 1 x h
            cla_kv = torch.stack(attention_list, dim=0).view(num_lang, -1, embed_dim).transpose(0, 1)  # length*lang*batch x lang x h
            cla_k = self.cla_linear_k(cla_kv)  # length*lang*batch x lang x h
            cla_v = self.cla_linear_v(cla_kv)  # length*lang*batch x lang x h
            cla_weight = torch.bmm(cla_q, cla_k.transpose(1, 2))  # length*lang*batch x 1 x lang
            cla_weight = F.softmax(cla_weight.float(), dim=-1).type_as(cla_weight)  # length*lang*batch x 1 x lang
            output = torch.bmm(cla_weight, cla_v).view(tgt_len, num_lang * bsz, embed_dim)  # length*lang*batch x 1 x h -> length x lang*batch x h
        elif Config.manner == "tanh":
            output = torch.tanh(sum(attention_list[1:])) * Config.tanh_weight + attention_list[0]
        else:
            raise Exception()
        return output, None

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

        if attn_mask is not None:
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

    def reorder_incremental_state(self, incremental_state, new_order):
        saved_state = incremental_state.get(self, {})
        saved_state['self_prev_key'] = saved_state['self_prev_key'].index_select(1, new_order)
        saved_state['self_prev_value'] = saved_state['self_prev_value'].index_select(1, new_order)
        saved_state['prev_key_padding_mask'] = saved_state['prev_key_padding_mask'].index_select(0, new_order)
