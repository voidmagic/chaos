import torch
from fairseq.data import LanguagePairDataset


class Dataset(LanguagePairDataset):
    def __getitem__(self, index):
        tgt_item = self.tgt[index]
        src_item = self.src[index]
        src_item = torch.cat((src_item, torch.fliplr(torch.unsqueeze(src_item, 0)).squeeze(0)))
        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
        }
        return example
