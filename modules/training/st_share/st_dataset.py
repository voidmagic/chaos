import logging
from pathlib import Path
from typing import Dict, List

import torch
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, SpeechToTextDataset, \
    SpeechToTextDatasetItem

logger = logging.getLogger(__name__)


class FastSpeechToTextDatasetCreator(SpeechToTextDatasetCreator):
    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[Dict],
            cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            n_frames_per_step,
            speaker_to_id,
    ) -> SpeechToTextDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        logger.info("Build FastSpeechToTextDataset")
        return FastSpeechToTextDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            speakers=speakers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            n_frames_per_step=n_frames_per_step,
            speaker_to_id=speaker_to_id,
        )


class FastSpeechToTextDataset(SpeechToTextDataset):
    def __getitem__(self, index: int) -> SpeechToTextDatasetItem:
        if self.is_train_split:
            source = torch.load("{}/item.{}".format("/tmp/must_data", index))
            # item = torch.load("{}/item.{}".format("/mnt/hdd/qwang/must_data", index))
        else:
            source = self._get_source_audio(index)
            source = self.pack_frames(source)

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=self.append_eos
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        speaker_id = None
        if self.speaker_to_id is not None:
            speaker_id = self.speaker_to_id[self.speakers[index]]
        return SpeechToTextDatasetItem(
            index=index, source=source, target=target, speaker_id=speaker_id
        )
