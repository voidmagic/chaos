import torch
from fairseq.data import FairseqDataset, data_utils


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    idx = torch.LongTensor([s["id"] for s in samples])

    src_tokens = data_utils.collate_tokens([s["source"] for s in samples], pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=False)
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    tgt_tokens = data_utils.collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=False)
    tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])

    prev_output_tokens = data_utils.collate_tokens([s["target"] for s in samples], pad_idx, eos_idx, left_pad=False, move_eos_to_beginning=True)

    _, sort_order = src_lengths.sort(descending=True)

    pair = torch.LongTensor([s["pair"] for s in samples])
    batch = {
        "id": idx.index_select(0, sort_order),
        "nsentences": len(samples),
        "ntokens": tgt_lengths.sum().item(),
        "net_input": {
            "src_tokens": src_tokens.index_select(0, sort_order),
            "src_lengths": src_lengths.index_select(0, sort_order),
            "prev_output_tokens": prev_output_tokens.index_select(0, sort_order)
        },
        "target": tgt_tokens.index_select(0, sort_order),
        "pair": pair.index_select(0, sort_order),
    }

    return batch


class MixedLanguagePairDataset(FairseqDataset):
    def __init__(self, bilingual_dataset, monolingual_dataset, ratio=2):
        self.bilingual_dataset = bilingual_dataset
        self.monolingual_dataset = monolingual_dataset
        self.ratio = ratio

    def __getitem__(self, idx):
        dataset, idx = self.get_dataset_and_index(idx)
        item = dataset[idx]  # {id, source, target}
        item['pair'] = dataset == self.bilingual_dataset  # is translation pair or not
        return item

    def __len__(self):
        return len(self.bilingual_dataset) * self.ratio + len(self.monolingual_dataset)

    def size(self, idx):
        dataset, idx = self.get_dataset_and_index(idx)
        return dataset.src_sizes[idx], dataset.tgt_sizes[idx]

    def num_tokens(self, idx):
        dataset, idx = self.get_dataset_and_index(idx)
        return max(dataset.src_sizes[idx], dataset.tgt_sizes[idx])

    def get_dataset_and_index(self, idx):
        if idx < len(self.monolingual_dataset):
            return self.monolingual_dataset, idx
        return self.bilingual_dataset, (idx - len(self.monolingual_dataset)) % len(self.bilingual_dataset)

    def collater(self, samples):
        return collate(samples, pad_idx=self.bilingual_dataset.src_dict.pad(), eos_idx=self.bilingual_dataset.eos)
