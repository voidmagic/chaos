from collections import OrderedDict
from typing import Dict

from fairseq.data import EpochBatchIterator, CountingIterator


class MyEpochBatchIterator:
    def __init__(self, iterators):
        self.iterators: Dict[str, EpochBatchIterator] = iterators
        self.it = None

    @property
    def first_batch(self):
        return {
            key: value.first_batch
            for key, value in self.iterators.items()
        }

    @property
    def epoch(self):
        return min([it.epoch for it in self.iterators.values()])

    @property
    def next_epoch_idx(self):
        return self.epoch + 1

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        self.it = BatchIterator({
            key: value.next_epoch_itr(shuffle=shuffle, fix_batches_to_gpus=fix_batches_to_gpus)
            for key, value in self.iterators.items()
        }, self.iterators)
        return self.it

    def end_of_epoch(self):
        return self.it.has_next()

    def state_dict(self):
        return {
            key: value.state_dict()
            for key, value in self.iterators.items()
        }


class BatchIterator:
    def __init__(self, iterators, epoch_iter):
        self.iterators = OrderedDict(iterators)
        self.epoch_iter = epoch_iter
        self.n = 0

    def __len__(self):
        return max([len(iterator) for iterator in self.iterators.values()])

    def __iter__(self):
        while True:
            self.n += 1
            rt = {}
            for key in self.iterators.keys():
                if not self.iterators[key].has_next():
                    self.iterators[key] = self.epoch_iter[key].next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
                rt[key] = next(self.iterators[key])
            yield rt
            if self.n >= max([len(iterator) for iterator in self.iterators.values()]):
                break

    def has_next(self):
        return self.n < len(self)
