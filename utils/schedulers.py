import logging
import math

import torch.optim as optim
from torch.optim import lr_scheduler as scheduler
from torch.optim.lr_scheduler import LambdaLR

log = logging.getLogger(__name__)


def make_scheduler(optimizer, config, double=False, teacher=False):
    warmup = config.WARMUP_EPOCHS
    if double:
        if teacher:
            total_iters = config.t_EPOCHS
        else:
            total_iters = config.s_EPOCHS
    else:
        total_iters = config.EPOCHS
    if config.SCHEDULER == "cosine":
        lr_scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=warmup, t_total=total_iters
        )
    elif config.SCHEDULER == "one_warmup_epoch":
        lr_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: config.WARMUP_LR / config.LR if epoch == 0 else 1,
        )
    else:
        lr_scheduler = scheduler.StepLR(
            optimizer, step_size=config.STEP_SIZE, gamma=0.1
        )
    return lr_scheduler


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Linearly increases learning rate from 0 to 1 over `warmup_steps`.
    Decreases learning rate from 1. to 0. over remaining
        `t_total - warmup_steps` steps following a cosine curve.
    If `cycles` (default=0.5) is different from default, learning rate
        follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=0.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        # log.info(f"vars: {self.warmup_steps}, {self.t_total}")
        super(WarmupCosineSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch, verbose=True
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            # log.info(f"STEP: {step}, LR1: {float(step) / float(max(1.0, self.warmup_steps))}")
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(
            max(1, self.t_total - self.warmup_steps)
        )
        # log.info(f"STEP: {step}, LR2: {max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))}")
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress))
        )
