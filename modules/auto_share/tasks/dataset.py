from fairseq.data.multi_corpus_dataset import MultiCorpusDataset


class MultilingualDataset(MultiCorpusDataset):


    def prefetch(self, indices):
        pass

