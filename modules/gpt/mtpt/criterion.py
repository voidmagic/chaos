from fairseq import utils
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
import torch.nn.functional as F
from fairseq.criterions import register_criterion


@register_criterion('pretrain_task_criterion')
class LabelSmoothedCrossEntropyWithTask(LabelSmoothedCrossEntropyCriterion):
    def forward(self, model, sample, reduce=True):
        net_output, class_output = model(**sample["net_input"], return_classification=True)
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (sample["target"].size(0) if self.sentence_avg else sample["ntokens"])
        class_output = utils.log_softmax(class_output, dim=-1)
        if 'pair' in sample:
            classify_loss = F.cross_entropy(class_output, sample['pair'])
        else:
            classify_loss = class_output.sum() * 0.

        loss = classify_loss + loss

        logging_output = {
            ""
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

