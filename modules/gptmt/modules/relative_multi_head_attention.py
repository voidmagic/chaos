from typing import Dict, Optional, Tuple

import torch
from fairseq import utils
from fairseq.modules import MultiheadAttention
from torch import Tensor


class RelativeMultiHeadAttention(MultiheadAttention):

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
        relative_position_embedding=None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        assert relative_position_embedding is not None
        relative_position_embedding_k, relative_position_embedding_v = relative_position_embedding

        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
        else:
            saved_state = None

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)
        q *= self.scaling

        q = (
            q.contiguous()
            .view(tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        k = (
            k.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        v = (
            v.contiguous()
            .view(-1, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        # 两个sequence_length的维度可以互换，区别是位置i(q)和位置j(kv)的相对位置是i-j还是j-i，这里使用j-i，和RelativePE文中一致
        relative_position_embedding_k = (
            relative_position_embedding_k   # [batch_size x sequence_length x sequence_length x hidden_dim]
            .transpose(0, 1).contiguous()   # [sequence_length x batch_size x sequence_length x hidden_dim]
            .transpose(1, 2).contiguous()   # [sequence_length x sequence_length x batch_size x hidden_dim]
            .view(tgt_len, tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(1, 2).contiguous()   # [sequence_length x batch_size x sequence_length x hidden_dim]
        )
        relative_position_embedding_v = (
            relative_position_embedding_v  # [batch_size x sequence_length x sequence_length x hidden_dim]
            .transpose(0, 1).contiguous()  # [sequence_length x batch_size x sequence_length x hidden_dim]
            .transpose(1, 2).contiguous()  # [sequence_length x sequence_length x batch_size x hidden_dim]
            .view(tgt_len, tgt_len, bsz * self.num_heads, self.head_dim)
            .transpose(1, 2).contiguous()  # [sequence_length x batch_size x sequence_length x hidden_dim]
        )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)
        assert k is not None
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        # relative_position_embedding: [sequence_length x batch_size x sequence_length x hidden_dim]
        # q: [batch_size x sequence_length x hidden_dim]
        # k: [batch_size x sequence_length x hidden_dim]
        attn_weights_all = []
        for i in range(q.size(1)):
            q_i = q[:, i:i+1]                                    # [batch_size x 1 x hidden_dim]
            k_respect_to_i = k + relative_position_embedding_k[i]  # [batch_size x sequence_length x hidden_dim]
            attn_weights = torch.bmm(q_i, k_respect_to_i.transpose(1, 2))  # [batch_size x 1 x sequence_length]
            attn_weights_all.append(attn_weights)

        attn_weights = torch.cat(attn_weights_all, dim=1)    # [batch_size*num_heads x sequence_length x sequence_length]
        # attn_weights = torch.bmm(q, k.transpose(1, 2))   # original attn_weights

        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        # attn_probs: [batch_size x sequence_length x sequence_length]
        # v:          [batch_size x sequence_length x hidden_dim]
        attn_all = []
        for i in range(attn_probs.size(1)):
            prob_i = attn_probs[:, i:i+1]                        # [batch_size x 1 x sequence_length]
            v_respect_to_i = v + relative_position_embedding_v[i]  # [batch_size x sequence_length x hidden_dim]
            attn = torch.bmm(prob_i, v_respect_to_i)             # [batch_size x 1 x hidden_dim]
            attn_all.append(attn)
        attn = torch.cat(attn_all, dim=1)
        # attn = torch.bmm(attn_probs, v)  # origin

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights

