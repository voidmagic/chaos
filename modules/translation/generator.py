import math
from typing import List, Dict, Optional

import torch
from fairseq.sequence_generator import SequenceGenerator as FairGenerator
from torch import Tensor


class SequenceGenerator(FairGenerator):
    def _generate(self, sample, *args, **kwargs):
        incremental_states = [{} for _ in self.model.models]
        net_input = sample["net_input"]
        src_tokens = net_input["src_tokens"]
        batch_size = src_tokens.size(0)
        beam_size = self.beam_size
        max_len = self.model.max_decoder_positions() - 1
        encoder_outs = self.model.forward_encoder(net_input)

        new_order = torch.arange(batch_size).repeat_interleave(beam_size).to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)

        scores = torch.zeros(batch_size * beam_size, max_len + 2).to(src_tokens).float()
        tokens = torch.zeros(batch_size * beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad)
        tokens[:, 0] = self.eos

        cands_to_ignore = torch.zeros(batch_size, beam_size).to(src_tokens).eq(-1)

        finalized = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]

        num_remaining_sent = batch_size
        bbsz_offsets = (torch.arange(0, batch_size) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, 2 * beam_size).type_as(tokens)

        for step in range(max_len + 1):
            lprobs, _ = self.model.forward_decoder(tokens[:, :step + 1], encoder_outs, incremental_states)
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            if step == max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos+1:] = -math.inf

            lprobs = lprobs + scores[:, step: step + 1]

            if step == 0:
                lprobs = lprobs[::beam_size]

            cand_scores, cand_indices = torch.topk(lprobs.view(batch_size, -1), k=beam_size * 2)
            cand_beams = torch.div(cand_indices, self.vocab_size, rounding_mode='floor')
            cand_indices = cand_indices.fmod(self.vocab_size)

            cand_bbsz_idx = cand_beams + bbsz_offsets

            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            batch_idxs = None
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, max_len)
                num_remaining_sent -= len(finalized_sents)

                if num_remaining_sent == 0:
                    break

                if len(finalized_sents) > 0:
                    new_bsz = batch_size - len(finalized_sents)

                    batch_mask = torch.ones(batch_size, dtype=torch.bool, device=cand_indices.device)
                    batch_mask[finalized_sents] = False
                    batch_idxs = torch.arange(batch_size, device=cand_indices.device).masked_select(batch_mask)

                    eos_mask = eos_mask[batch_idxs]
                    cand_beams = cand_beams[batch_idxs]
                    bbsz_offsets.resize_(new_bsz, 1)
                    cand_bbsz_idx = cand_beams + bbsz_offsets
                    cand_scores = cand_scores[batch_idxs]
                    cand_indices = cand_indices[batch_idxs]

                    cands_to_ignore = cands_to_ignore[batch_idxs]

                    scores = scores.view(batch_size, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                    tokens = tokens.view(batch_size, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                    batch_size = new_bsz

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = eos_mask.type_as(cand_offsets) * beam_size * 2 + cand_offsets[: eos_mask.size(1)]
            new_cands_to_ignore, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)
            cands_to_ignore = new_cands_to_ignore.ge(2 * beam_size)[:, :beam_size]
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos).view(-1)

            tokens[:, :step+1] = torch.index_select(tokens[:, :step+1], dim=0, index=active_bbsz_idx)
            tokens.view(batch_size, beam_size, -1)[:, :, step+1] = torch.gather(cand_indices, dim=1, index=active_hypos)

            if step > 0:
                scores[:, :step+1] = torch.index_select(scores[:, :step+1], dim=0, index=active_bbsz_idx)

            scores.view(batch_size, beam_size, -1)[:, :, step+1] = torch.gather(cand_scores, dim=1, index=active_hypos)

            if active_bbsz_idx is not None:
                if batch_idxs is not None:
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    active_bbsz_idx.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                self.model.reorder_incremental_state(incremental_states, active_bbsz_idx)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, active_bbsz_idx)

        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        return finalized


    def finalize_hypos(
        self,
        step: int,
        bbsz_idx: torch.Tensor,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        max_len: int,
        *args, **kwargs
    ):
        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[:, 1: step + 2]  # skip the first index, which is EOS
        tokens_clone[:, step] = self.eos

        # normalize sentence-level scores
        eos_scores = eos_scores / (step + 1) ** self.len_penalty

        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = torch.div(idx, beam_size, rounding_mode='floor')
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # print(f"{step} FINISHED {idx} {score} {sent}={unfin_idx} {cum_unfin}")
            seen = str(int(sent)) + "_" + str(int(unfin_idx))
            if seen not in sents_seen:
                sents_seen[seen] = None

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                finalized[sent].append({"tokens": tokens_clone[i], "score": score, "alignment": None})

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

