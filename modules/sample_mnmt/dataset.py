from collections import OrderedDict
from itertools import chain, cycle
import random

from fairseq.data import RoundRobinZipDatasets


class MultilingualSampledDataset(RoundRobinZipDatasets):
    def __init__(self, datasets, eval_key=None, sample_method='temperature', temperature=5):
        super(MultilingualSampledDataset, self).__init__(datasets, eval_key)
        self.sample_method = sample_method
        self.temperature = temperature

    def map_multilingual_index(self, index):
        for key, dataset in self.datasets.items():
            if index < len(dataset):
                return key, index
            index -= len(dataset)
        raise IndexError(
            "index {} out of range {}".format(index, sum(len(dataset for dataset in self.datasets.values()))))

    def __getitem__(self, index):
        key, index = self.map_multilingual_index(index)
        return key, self.datasets[key][index]

    def collater(self, samples):
        if len(samples) == 0:
            return None
        assert len(set([sample[0] for sample in samples])) == 1, "All samples in a batch must be in the same language."
        current_key = samples[0][0]
        samples = [sample[1] for sample in samples]
        batch_dict = OrderedDict([
            (key, dataset.collater(samples) if key == current_key else None) for key, dataset in self.datasets.items()
        ])
        return batch_dict if self.eval_key is None else batch_dict[self.eval_key]

    def ordered_indices(self):
        super(MultilingualSampledDataset, self).ordered_indices()
        return self._ordered_indices

    def filter_indices_by_size(self, indices, max_sizes):
        filtered_ignored = OrderedDict([
            (key, dataset.filter_indices_by_size(indices[key], max_sizes[key]))
            for key, dataset in self.datasets.items()
        ])
        filtered = OrderedDict([(key, value[0]) for key, value in filtered_ignored.items()])
        ignored = OrderedDict([(key, value[1]) for key, value in filtered_ignored.items()])
        return filtered, list(chain.from_iterable(ignored.values()))

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        batch_sampler = OrderedDict([
            (key, dataset.batch_by_size(indices[key], max_tokens, max_sentences, required_batch_size_multiple))
            for key, dataset in self.datasets.items()
        ])
        batch_sampler = self.map_multilingual_sampler(batch_sampler)
        return batch_sampler

    def map_multilingual_sampler(self, batch_sampler_dict):
        assert self.sample_method in ['uniform', 'temperature', 'proportional']
        if self.sample_method == 'uniform':
            weights = OrderedDict([(key, 1) for key, value in self.datasets.items()])
        elif self.sample_method == 'proportional':
            weights = {key: len(value) for key, value in batch_sampler_dict.items()}
        else:
            weights = {key: len(value) ** (1 / self.temperature) for key, value in batch_sampler_dict.items()}

        offset = [(key, len(dataset)) for key, dataset in self.datasets.items()]
        offset = [(key, sum([v[1] for v in offset[:i + 1]]) - value) for i, (key, value) in enumerate(offset)]
        offset = OrderedDict(offset)
        for key, dataset in self.datasets.items():
            batch_sampler = batch_sampler_dict[key]
            for row in range(len(batch_sampler)):
                for col in range(len(batch_sampler[row])):
                    batch_sampler[row][col] += offset[key]
            random.shuffle(batch_sampler_dict[key])
        max_key, _ = max(batch_sampler_dict.items(), key=lambda item: len(item[1]))
        estimate_batch = {key: int(weight / weights[max_key] * len(batch_sampler_dict[max_key]))
                          for key, weight in weights.items()}
        endless_cycle = {key: cycle(batch_sampler) for key, batch_sampler in batch_sampler_dict.items()}
        batch_sampler_weighted = [[next(batch_iter) for _ in range(estimate_batch[key])]
                                  for key, batch_iter in endless_cycle.items()]
        return list(chain.from_iterable(batch_sampler_weighted))
