import random
import logging
from collections import OrderedDict
from typing import Dict

from fairseq.data import EpochBatchIterator

logger = logging.getLogger(__name__)


class MyEpochBatchIterator:
    def __init__(self, iterators, sample_prop):
        self.iterators: Dict[str, EpochBatchIterator] = iterators
        self.sample_prop = sample_prop
        self.it = BatchIterator({
            key: value.next_epoch_itr(shuffle=True, fix_batches_to_gpus=False)
            for key, value in self.iterators.items()
        }, self.iterators, self.sample_prop)
        self.epoch = 0

    @property
    def first_batch(self):
        return {
            key: value.first_batch
            for key, value in self.iterators.items()
        }

    @property
    def next_epoch_idx(self):
        return self.epoch + 1

    def next_epoch_itr(self, *args, **kwargs):
        self.epoch += 1
        self.it.reset()
        return self.it

    def end_of_epoch(self):
        return self.it.has_next()

    def state_dict(self):
        return {
            key: value.state_dict()
            for key, value in self.iterators.items()
        }

    def load_state_dict(self, stat_dict):
        pass


class BatchIterator:
    def __init__(self, iterators, epoch_iter, sample_prop):
        self.iterators = OrderedDict(iterators)
        self.epoch_iter = epoch_iter
        self.sample_prop = sample_prop
        self.n = 0
        logger.info('Number of batches: ')
        for lang_pair, it in sorted(self.iterators.items(), key=lambda p: -len(p[1])):
            logger.info('{}: {} - {}'.format(lang_pair, len(it), int(len(it) / self.sample_prop[lang_pair])))

    def __len__(self):
        return sum([int(len(value) / self.sample_prop[key]) for key, value in self.iterators.items()])

    def __iter__(self):
        while True:
            # 随机挑选一种语言，采样一个batch
            self.n += 1
            lang_pairs = list(self.sample_prop.keys())
            sample_prop_normalized = [self.sample_prop[lang_pair] / sum(self.sample_prop.values()) for lang_pair in lang_pairs]
            lang_pair = random.choices(lang_pairs, weights=sample_prop_normalized)[0]
            if not self.iterators[lang_pair].has_next():
                self.iterators[lang_pair] = self.epoch_iter[lang_pair].next_epoch_itr(
                    shuffle=True, fix_batches_to_gpus=False)
            rt = {
                key: next(self.iterators[key]) if key == lang_pair else None
                for key in self.iterators.keys()
            }
            yield rt
            if self.n >= len(self):
                break

    def has_next(self):
        return self.n < len(self)

    def reset(self):
        self.n = 0