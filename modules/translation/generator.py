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
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size
        max_len = self.model.max_decoder_positions() - 1
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()
        tokens = torch.zeros(bsz * beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad)
        tokens[:, 0] = self.eos

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)

        # list of completed sentences
        finalized = [[] for _ in range(bsz)]

        # a boolean array indicating if the sentence at the index is finished or not
        finished = [False for i in range(bsz)]

        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)

            lprobs, _ = self.model.forward_decoder(tokens[:, : step + 1], encoder_outs, incremental_states)
            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos+1:] = -math.inf

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = beam_search(step, lprobs.view(bsz, -1, self.vocab_size), scores.view(bsz, beam_size, -1)[:, :, :step])

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, max_len)
                num_remaining_sent -= len(finalized_sents)

            if num_remaining_sent == 0:
                break

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                cands_to_ignore = cands_to_ignore[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[: eos_mask.size(1)])

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(active_mask, k=beam_size, dim=1, largest=False)

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos).view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(tokens[:, : step + 1], dim=0, index=active_bbsz_idx)
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)

            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        return finalized


    def finalize_hypos(
        self,
        step: int,
        bbsz_idx,
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

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, : step + 1]
        pos_scores[:, step] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        eos_scores /= (step + 1) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # set() is not supported in script export

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # print(f"{step} FINISHED {idx} {score} {sent}={unfin_idx} {cum_unfin}")
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None

            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": torch.empty(0),  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(step, unfin_idx, max_len, len(finalized[sent]), beam_size):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished




def beam_search(step: int, lprobs, scores: Optional[Tensor]):
    bsz, beam_size, vocab_size = lprobs.size()

    if step == 0:
        # at the first step all hypotheses are equally likely, so use
        # only the first beam
        lprobs = lprobs[:, ::beam_size, :].contiguous()
    else:
        # make probs contain cumulative scores for each hypothesis
        assert scores is not None
        lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)

    top_prediction = torch.topk(
        lprobs.view(bsz, -1),
        k=min(
            # Take the best 2 x beam_size predictions. We'll choose the first
            # beam_size of these which don't predict eos to continue with.
            beam_size * 2,
            lprobs.view(bsz, -1).size(1) - 1,  # -1 so we never select pad
        ),
    )
    scores_buf = top_prediction[0]
    indices_buf = top_prediction[1]
    # Project back into relative indices and beams
    beams_buf = indices_buf // vocab_size
    indices_buf = indices_buf.fmod(vocab_size)

    # At this point, beams_buf and indices_buf are single-dim and contain relative indices
    return scores_buf, indices_buf, beams_buf
