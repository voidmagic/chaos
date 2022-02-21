import logging
import os
import socket
from collections import OrderedDict
from pathlib import Path

import torch
from fairseq.data import Dictionary
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask

from .st_dataset import FastSpeechToTextDatasetCreator
from ...basics.sample_mnmt.dataset import MultilingualSampledDataset

logger = logging.getLogger(__name__)


@register_task("speech_translation")
class SpeechTranslation(SpeechToTextTask):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        if not socket.gethostname() == "cip57":
            new_home = os.path.expanduser('~')
            self.data_cfg.config["vocab_filename"] = self.data_cfg.config["vocab_filename"].replace("/mnt/hdd/qwang", new_home)
            self.data_cfg.config["bpe_tokenizer"]["sentencepiece_model"] = self.data_cfg.config["bpe_tokenizer"]["sentencepiece_model"].replace("/mnt/hdd/qwang", new_home)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        is_valid_split = split.startswith("dev")
        if is_valid_split:
            split = split.replace(":", ",")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = FastSpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            speaker_to_id=self.speaker_to_id,
        )

        if is_train_split or is_valid_split:
            dataset_dict = OrderedDict([(dataset.split.split("_")[1], dataset) for dataset in self.datasets[split].datasets])
            self.datasets[split] = MultilingualSampledDataset(dataset_dict)
            if is_valid_split:
                self.datasets[split.replace(",", ":")] = self.datasets[split]

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)
        if not socket.gethostname() == "cip57":
            new_home = os.path.expanduser('~')
            data_cfg.config["vocab_filename"] = data_cfg.config["vocab_filename"].replace("/mnt/hdd/qwang", new_home)
            data_cfg.config["bpe_tokenizer"]["sentencepiece_model"] = data_cfg.config["bpe_tokenizer"]["sentencepiece_model"].replace("/mnt/hdd/qwang", new_home)
        dict_path = Path(args.data) / data_cfg.vocab_filename
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}")
        return cls(args, tgt_dict)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        for model_key in sample.keys():
            if sample[model_key] is not None:
                loss, sample_size, logging_output = criterion(model.models[model_key], sample[model_key])
                if ignore_grad:
                    loss *= 0
                optimizer.backward(loss)
                return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            for model_key in sample.keys():
                if sample[model_key] is not None:
                    loss, sample_size, logging_output = criterion(model.models[model_key], sample[model_key])
                    return loss, sample_size, logging_output
