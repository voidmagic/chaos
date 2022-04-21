from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask


class PartialMultilingualDatasetManager(MultilingualDatasetManager):
    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return PartialMultilingualDatasetManager(args, lang_pairs, langs, dicts, sampling_method)

    def get_data_paths_and_lang_pairs(self, split):
        datapaths = {"main": self.args.data}
        lang_pairs = {"main": self.args.eval_lang_pairs if split == self.args.valid_subset else self.lang_pairs}
        return datapaths, lang_pairs

    def get_train_sampling_ratios(self, data_param_list, datasets, epoch=1, shard_epoch=None):
        data_sizes = self.get_train_dataset_sizes(data_param_list, datasets, epoch, shard_epoch)
        sampling_func = self.sampling_method.sampling_method_selector()
        sample_ratios = sampling_func(data_sizes) if sampling_func is not None else None
        for i in range(len(sample_ratios)):
            if self.lang_pairs[i] not in self.args.eval_lang_pairs:
                # 这是个辅助语言, rate decay
                sample_ratios[i] = sample_ratios[i] * (1 / (1 + 0.5 * (epoch - 1))) / len(self.lang_pairs) * len(self.args.eval_lang_pairs)
        return sample_ratios

    def has_sharded_data(self, split):
        return True  # reload data each epoch, to reinitialize sampling


@register_task("aux_task")
class AuxTask(TranslationMultiSimpleEpochTask):

    def __init__(self, args, langs, dicts, training):
        super(AuxTask, self).__init__(args, langs, dicts, training)

        if isinstance(args.eval_lang_pairs, str):
            args.eval_lang_pairs = args.eval_lang_pairs.split(",")
        self.data_manager = PartialMultilingualDatasetManager.setup_data_manager(args, self.lang_pairs, langs, dicts, self.sampling_method)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--eval-lang-pairs', default=None, metavar='PAIRS', help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        TranslationMultiSimpleEpochTask.add_args(parser)
