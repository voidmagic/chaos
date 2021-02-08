from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq.tasks import register_task


@register_task('inter_nat_task')
class InterNatTask(TranslationLevenshteinTask):
    pass
