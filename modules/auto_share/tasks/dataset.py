from collections import OrderedDict
import os
import numpy as np
import itertools
from fairseq.data.round_robin_zip_datasets import RoundRobinZipDatasets


class TemperatureRoundRobinDataset(RoundRobinZipDatasets):
    def __init__(self, *args, **kwargs):
        super(TemperatureRoundRobinDataset, self).__init__(*args, **kwargs)
        self.sample_method = os.environ.get('SAMPLE_METHOD', 'uniform')
        self.sample_temperature = int(os.environ.get('SAMPLE_TEMP', '5'))
        sorted_by_size = sorted(self.datasets.values(), key=lambda dataset: -len(dataset))
        self.sizes_for_filter = np.array(list(zip(sorted_by_size[0].sizes, *[itertools.cycle(dataset.sizes) for dataset in sorted_by_size[1:]]))).reshape(len(sorted_by_size[0]), -1).max(axis=1)

    def _map_index(self, key, index):
        assert (
                self._ordered_indices is not None
        ), "Must call RoundRobinZipDatasets.ordered_indices() first"
        return self._ordered_indices[key][index % len(self.datasets[key])]

    def __getitem__(self, index):
        if self.eval_key is None:
            sample_dict = OrderedDict(
                [
                    (key, dataset[self._map_index(key, index)])
                    for key, dataset in self.datasets.items()
                ]
            )
            return self.multilingual_sampler(sample_dict, index)
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]

    def multilingual_sampler(self, sample_dict, index):
        if self.sample_method == 'uniform':
            return sample_dict
        elif self.sample_method == 'temperature':
            return self.temperature_based_sampler(sample_dict)
        elif self.sample_method == 'proportional':
            return self.proportional_sampler(sample_dict, index)
        else:
            raise NotImplementedError('sample method is not implemented')

    def temperature_based_sampler(self, sample_dict):
        return sample_dict

    def proportional_sampler(self, sample_dict, index):
        # for key, sample in sample_dict.items():
        #     if index // len(self.datasets[key]) > 0:
        #         sample_dict[key] = None
        #     if key == 'en-br':
        #         sample_dict[key] = None
        return sample_dict

    def num_tokens(self, index):
        return self.sizes_for_filter[index]

    def filter_indices_by_size(self, indices, max_sizes):
        ignored = indices[self.sizes_for_filter[indices] > 512].tolist()
        indices = indices[self.sizes_for_filter[indices] <= 512]
        return indices, ignored
