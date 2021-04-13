import numpy as np
import itertools
from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets


class FastRoundRobinDataset(RoundRobinZipDatasets):
    def __init__(self, *args, **kwargs):
        super(FastRoundRobinDataset, self).__init__(*args, **kwargs)
        sorted_by_size = sorted(self.datasets.values(), key=lambda dataset: -len(dataset))
        sizes = np.array(list(zip(sorted_by_size[0].sizes, *[itertools.cycle(dataset.sizes) for dataset in sorted_by_size[1:]])))
        self.sizes_for_filter = sizes.reshape(len(sorted_by_size[0]), -1).max(axis=1)
        self.sizes_for_batch = self.sizes_for_filter

    def num_tokens(self, index):
        return self.sizes_for_batch[index]

    def filter_indices_by_size(self, indices, max_sizes):
        ignored = indices[self.sizes_for_filter[indices] > 512].tolist()
        indices = indices[self.sizes_for_filter[indices] <= 512]
        return indices, ignored
