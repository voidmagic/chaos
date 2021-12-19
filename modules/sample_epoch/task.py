
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask


@register_task("simple_epoch")
class MyTranslationMultiSimpleEpochTask(TranslationMultiSimpleEpochTask):
    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        # setattr(args, 'keep_inference_langtok', True)
        return super().build_generator(models, self.args, seq_gen_cls, extra_gen_cls_kwargs)
