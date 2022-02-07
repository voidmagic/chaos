import logging

from fairseq.data import RoundRobinZipDatasets
import numpy as np


logger = logging.getLogger(__name__)


class RoundRobinSTDataset(RoundRobinZipDatasets):
    def filter_indices_by_size(self, indices, max_positions=None):
        if not isinstance(max_positions, dict):
            max_positions = {k: max_positions for k in self.datasets.keys()}
        ignored_some = False
        for key, dataset in self.datasets.items():
            self._ordered_indices[key], ignored = dataset.filter_indices_by_size(
                self._ordered_indices[key], max_positions[key]
            )
            if len(ignored) > 0:
                ignored_some = True
                logger.warning(
                    f"{len(ignored)} samples from {key} have invalid sizes and will be skipped, "
                    f"max_positions={max_positions[key]}, first few sample ids={ignored[:10]}"
                )
        return np.arange(len(self)), [0] if ignored_some else []

