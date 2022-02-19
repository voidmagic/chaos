import collections
import os
import torch
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask

from .st_dataset import FastSpeechToTextDatasetCreator


@register_task("basic_speech_translation")
class BasicSpeechTranslation(SpeechToTextTask):

    gradients = collections.defaultdict(list)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
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

    def valid_step(self, sample, model, criterion):
        model.eval()
        model.zero_grad()
        loss, sample_size, logging_output = criterion(model, sample)
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.gradients[name].append(p.grad)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super(BasicSpeechTranslation, self).reduce_metrics(logging_outputs, criterion)
        filename = os.environ.get("gradient_path", default="{}.pt".format(list(self.datasets.keys())[0]))
        torch.save({key: torch.mean(torch.stack(value, dim=0), dim=0).view(-1).cpu() for key, value in self.gradients.items()}, filename)
