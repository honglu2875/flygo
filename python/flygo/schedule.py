"""Host-side learning rates indexed by the absolute optimizer update."""
from dataclasses import asdict,dataclass
import math


@dataclass(frozen=True)
class Schedule:
    peak:float
    warmup_steps:int=0
    decay_until:int=0
    final_ratio:float=.1

    def __post_init__(self):
        if not math.isfinite(self.peak) or self.peak<=0:
            raise ValueError('Schedule peak must be finite and positive')
        if min(self.warmup_steps,self.decay_until)<0:
            raise ValueError('Schedule update counts must be nonnegative')
        if self.decay_until and self.decay_until<=max(1,self.warmup_steps):
            raise ValueError('Cosine endpoint must follow the peak-rate update')
        if not math.isfinite(self.final_ratio) or not 0<self.final_ratio<=1:
            raise ValueError('Final learning-rate ratio must be in (0,1]')

    def rate(self,update):
        if update<1:raise ValueError('Optimizer updates are one-based')
        if update<=self.warmup_steps:return self.peak*update/self.warmup_steps
        if not self.decay_until:return self.peak
        start=max(1,self.warmup_steps)
        fraction=min(1,max(0,(update-start)/(self.decay_until-start)))
        return self.peak*(self.final_ratio+(1-self.final_ratio)*(1+math.cos(math.pi*fraction))/2)

    def contract(self):
        return dict(version='absolute-update-rate-v1',**asdict(self))

    def check_resume(self,training_contract):
        previous=training_contract.get('schedule')
        if previous is None:
            if self.warmup_steps or self.decay_until:
                raise ValueError('Legacy constant-rate checkpoint has no schedule to resume')
        elif previous!=self.contract():
            raise ValueError('Resume learning-rate schedule differs from the checkpoint')
