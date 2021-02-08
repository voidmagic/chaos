from fairseq.data import LanguagePairDataset, data_utils
import torch


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    idx = torch.LongTensor([s["id"] for s in samples])

    src_tokens = data_utils.collate_tokens([s["source"] for s in samples], pad_idx, eos_idx, left_pad=True, move_eos_to_beginning=False)
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    target = data_utils.collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=False)
    tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])

    prev_output_tokens = data_utils.collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=True)

    _, sort_order = src_lengths.sort(descending=True)

    batch = {
        "id": idx.index_select(0, sort_order),
        "nsentences": len(samples),
        "ntokens": tgt_lengths.sum().item(),
        "net_input": {
            "src_tokens": src_tokens.index_select(0, sort_order),
            "src_lengths": src_lengths.index_select(0, sort_order),
            "prev_output_tokens": prev_output_tokens.index_select(0, sort_order)
        },
        "target": target.index_select(0, sort_order),
    }

    return batch


class Dataset(LanguagePairDataset):
    def __init__(self, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)

    def collater(self, samples, pad_to_length=None):
        res = collate(samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
        return res
