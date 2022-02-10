from collections import OrderedDict, defaultdict

import torch
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from .round_robin_st_dataset import RoundRobinSTDataset


@register_task("speech_translation")
class SpeechTranslation(SpeechToTextTask):
    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        is_valid_split = split.startswith("dev")
        if is_valid_split:
            split = split.replace(":", ",")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
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
            self.datasets[split] = RoundRobinSTDataset(dataset_dict)
            if is_valid_split:
                self.datasets[split.replace(",", ":")] = self.datasets[split]

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
        for model_key in sample.keys():
            loss, sample_size, logging_output = criterion(model.models[model_key], sample[model_key])
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            for k in logging_output:
                agg_logging_output[k] += logging_output[k]
                agg_logging_output[f"{model_key}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0.0, 0.0, defaultdict(float)
            for model_key in sample.keys():
                loss, sample_size, logging_output = criterion(model.models[model_key], sample[model_key])
                agg_loss += loss.data.item()
                agg_sample_size += sample_size
                for k in logging_output:
                    agg_logging_output[k] += logging_output[k]
                    agg_logging_output[f"{model_key}:{k}"] += logging_output[k]
        return agg_loss, agg_sample_size, agg_logging_output
