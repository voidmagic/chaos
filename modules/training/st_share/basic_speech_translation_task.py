from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask

from .st_dataset import FastSpeechToTextDatasetCreator


@register_task("basic_speech_translation")
class BasicSpeechTranslation(SpeechToTextTask):
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
