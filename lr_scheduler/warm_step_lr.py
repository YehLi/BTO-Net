import torch
from lib.config import cfg

class WarmStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        gamma,
        step_size,
        decay_start,
        last_epoch=-1,
    ):

        self.gamma = gamma
        self.step_size = step_size
        self.decay_start = decay_start
        super(WarmStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch + 1 < self.decay_start:
            ret = [min(base_lr, (self.last_epoch+1)*0.0001) for base_lr in self.base_lrs]
            return ret
        else:
            ret = [min(base_lr, (self.last_epoch+1)*0.0001) * self.gamma ** ((self.last_epoch + 1 - self.decay_start)//self.step_size) for base_lr in self.base_lrs]
            return ret