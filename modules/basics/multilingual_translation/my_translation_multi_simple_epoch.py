import datetime
import logging

import torch
from fairseq.data import data_utils
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.optim import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.utils import FileContentsAction

logger = logging.getLogger(__name__)


def get_time_gap(s, e):
    return (datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)).__str__()


@register_task("my_translation_multi_simple_epoch")
class MyTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr',
                            action=FileContentsAction)
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')

        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(cls.load_dictionary, args, **kwargs)
        return cls(args, langs, dicts, training)

    def build_model(self, args):
        return super().build_model(args)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):

        def reduce_logging_output(_logging_output):
            return {
                key: value.sum() if isinstance(value, torch.Tensor) else value
                for key, value in _logging_output.items()
            }

        model.train()
        model.set_num_updates(update_num)
        loss, sample_size, logging_output = criterion(model, sample, reduce=False)
        loss = loss.sum()
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, reduce_logging_output(logging_output)

    def infer_task(self, sample):
        source_tokens = sample["net_input"]["src_tokens"]
