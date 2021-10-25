
import math
from typing import Optional

import torch
from fairseq.sequence_generator import SequenceGenerator as FairSequenceGenerator
from torch import Tensor


class SequenceGenerator(FairSequenceGenerator):

    def _generate(self, sample, prefix_tokens=None, constraints=None, bos_token=None):
        from modules.sync_mnmt.task import Config

        n_langs = Config.n_lang
        target_lang = Config.infer_target
        all_target_langs = list(Config.lang_idx)
        all_target_langs.remove(target_lang)
        all_target_langs.insert(0, target_lang)

        incremental_states = [{} for _ in self.model.models]
        net_input = sample["net_input"]

        src_tokens = net_input["src_tokens"]

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        max_len = min(int(self.max_len_a * src_len + self.max_len_b), self.model.max_decoder_positions() - 1)

        assert (self.min_len <= max_len), "min_len cannot be larger than max_len, please adjust these!"

        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).repeat(n_langs, 1).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)


        # initialize buffers
        tokens = []
        scores = []
        for lang_idx in all_target_langs:
            _tokens = torch.zeros(bsz * beam_size, max_len + 2).to(src_tokens).long().fill_(self.pad)  # +2 for eos and pad
            _tokens[:, 0] = lang_idx
            tokens.append(_tokens)
            _scores = torch.zeros(bsz * beam_size, max_len + 1).to(src_tokens).float()  # +1 for eos
            scores.append(_scores)
        tokens = torch.cat(tokens, dim=0)
        scores = torch.cat(scores, dim=0)

        bsz = bsz * n_langs

        # A list that indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then cands_to_ignore would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        cands_to_ignore = torch.zeros(bsz, beam_size).to(src_tokens).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for _ in range(bsz)]
        finished = [False for _ in range(bsz)]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        for step in range(max_len + 1):  # one extra step for EOS marker
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(encoder_outs, reorder_state)

            lprobs, _ = self.model.forward_decoder(tokens[:, :step+1], encoder_outs, incremental_states)

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # lang x batch x beam, vocab
            lprobs[tokens[:, 0] != target_lang, self.eos] = -math.inf  # never select eos for non-target languages

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            scores = scores.type_as(lprobs)

            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(step, lprobs.view(bsz, -1, self.vocab_size), scores.view(bsz, beam_size, -1)[:, :, :step])

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos. Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            eos_mask.view(n_langs, -1)[:] = torch.logical_or(eos_mask.view(n_langs, -1), eos_mask.view(n_langs, -1)[0])

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size])

            finalized_sents = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size])
                finalized_sents = self.finalize_hypos(step, eos_bbsz_idx, eos_scores, tokens, scores, finalized, finished, beam_size, None, None, max_len)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len

            # Remove finalized sentences (ones for which {beam_size} finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(bsz, dtype=torch.bool, device=cand_indices.device)
                batch_mask[finalized_sents] = False
                batch_idxs = torch.arange(bsz, device=cand_indices.device).masked_select(batch_mask)

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
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_bbsz_idx = active_bbsz_idx.view(-1)

            # copy tokens and scores for active hypotheses
            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step + 1] = torch.index_select(tokens[:, : step + 1], dim=0, index=active_bbsz_idx)
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step + 1] = torch.gather(cand_indices, dim=1, index=active_hypos)

            if step > 0:
                scores[:, :step] = torch.index_select(scores[:, :step], dim=0, index=active_bbsz_idx)

            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(cand_scores, dim=1, index=active_hypos)

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        assert len(finalized) % n_langs == 0

        for sent in range(len(finalized) // n_langs):
            scores = torch.tensor([float(elem["score"].item()) for elem in finalized[sent]])
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
        return finalized
