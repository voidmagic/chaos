import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask


@register_task("protocol_multi_task")
class ProtocolMultiTask(TranslationMultiSimpleEpochTask):
    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            target = sample['target']
            loss, sample_size, logging_output = criterion(model, sample, reduce=False)
            record_loss = loss.view(target.size())
            record_loss = torch.exp(-record_loss).cpu()

        for s_id, s_source, s_target, s_record_loss in zip(sample['id'], sample['net_input']['src_tokens'], target, record_loss):
            s_record_loss = " ".join([format(sla, ".4f") for sla in s_record_loss[s_target.ne(self.target_dictionary.pad())].tolist()])
            s_source = self.source_dictionary.string(s_source[s_source.ne(self.source_dictionary.pad())], include_eos=True)
            s_target = self.target_dictionary.string(s_target[s_target.ne(self.target_dictionary.pad())], include_eos=True)
            print("S-{}\t{}".format(s_id, s_source))
            print("T-{}\t{}".format(s_id, s_target))
            print("L-{}\t{}".format(s_id, s_record_loss))
        return loss.sum(), sample_size, {
            key: value.sum() if 'loss' in key else value for key, value in logging_output.items()
        }
