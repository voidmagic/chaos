import logging

from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask

from .dataset import MultilingualSampledDataset

logger = logging.getLogger(__name__)


@register_task("sample_mnmt")
class SampledMultilingualTask(MultilingualTranslationTask):
    def load_dataset(self, split, epoch=1, **kwargs):
        super(SampledMultilingualTask, self).load_dataset(split, epoch)
        self.datasets[split] = MultilingualSampledDataset(self.datasets[split].datasets, self.datasets[split].eval_key)
