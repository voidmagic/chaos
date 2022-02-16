from fairseq.optim import register_optimizer
from fairseq.optim.adam import FairseqAdam, FairseqAdamConfig


@register_optimizer("e_adam", dataclass=FairseqAdamConfig)
class EmptyAdam(FairseqAdam):
    def __init__(self, *args, **kwargs):
        super(EmptyAdam, self).__init__(*args, **kwargs)
        self.__class__.__name__ = "FP16Optimizer"

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        pass
