import logging
from pathlib import Path
from typing import Dict, List

from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDatasetCreator, SpeechToTextDataset

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
        audio_paths = [path_mapping(path) for path in audio_paths]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        speakers = [s.get(cls.KEY_SPEAKER, cls.DEFAULT_SPEAKER) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        logger.info("Build FastSpeechToTextDataset")
        return SpeechToTextDataset(
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


def path_mapping(original_path):
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-fr", "/data/tmp/en-fr")
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-it", "/data/tmp/en-it")
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-nl", "/tmp/en-nl")
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-pt", "/tmp/en-pt")
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-ro", "/home/supercip/mustc/en-ro")
    original_path = original_path.replace("/mnt/hdd/qwang/029-must/002-dataset/001-mustc/MUSTC/en-ru", "/home/supercip/mustc/en-ru")
    return original_path
