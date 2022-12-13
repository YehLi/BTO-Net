import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lib.config import cfg
import lr_scheduler
from optimizer.radam import RAdam, AdamW
from optimizer.adabelief import AdaBelief
import lr_scheduler.warmup_lr as warmup_lr

class Optimizer(nn.Module):
    def __init__(self, 
        model, 
        base_lr=0.0005, 
        min_lr=0.0000,
        step_per_epoch=0, 
        warmup_epoch=0, 
        max_epoch=0,
        lr_policy=None
    ):
        super(Optimizer, self).__init__()
        self.setup_optimizer(
            model, 
            base_lr=base_lr, 
            min_lr=min_lr,
            step_per_epoch=step_per_epoch, 
            warmup_epoch=warmup_epoch, 
            max_epoch=max_epoch, 
            lr_policy=lr_policy)

    def setup_optimizer_only(self, model, base_lr):
        params = []
        no_decay = ["bias", "layer_norm.bias", "layer_norm.weight"]
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            lr = base_lr
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if any(nd in key for nd in no_decay):
                weight_decay = 0

            if "bias" in key:
                lr = base_lr * cfg.SOLVER.BIAS_LR_FACTOR 
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        if cfg.SOLVER.TYPE == 'SGD':
            self.optimizer = torch.optim.SGD(
                params, 
                lr = base_lr, 
                momentum = cfg.SOLVER.SGD.MOMENTUM
            )
        elif cfg.SOLVER.TYPE == 'ADAM':
            self.optimizer = torch.optim.Adam(
                params,
                lr = base_lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAMAX':
            self.optimizer = torch.optim.Adamax(
                params,
                lr = base_lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADAGRAD':
            self.optimizer = torch.optim.Adagrad(
                params,
                lr = base_lr
            )
        elif cfg.SOLVER.TYPE == 'RMSPROP':
            self.optimizer = torch.optim.RMSprop(
                params, 
                lr = base_lr
            )
        elif cfg.SOLVER.TYPE == 'RADAM':
            self.optimizer = RAdam(
                params, 
                lr = base_lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        elif cfg.SOLVER.TYPE == 'ADABELIEF':
            self.optimizer = AdaBelief(
                params, 
                lr = base_lr, 
                betas = cfg.SOLVER.ADAM.BETAS, 
                eps = cfg.SOLVER.ADAM.EPS
            )
        else:
            raise NotImplementedError

    def setup_optimizer(self, model, base_lr, min_lr, step_per_epoch=0, warmup_epoch=0, max_epoch=0, lr_policy=None): 
        self.step_type = lr_policy.SETP_TYPE
        self.setup_optimizer_only(model, base_lr=base_lr)

        warmup_steps = step_per_epoch * warmup_epoch
        n_steps = step_per_epoch * max_epoch

        if lr_policy.TYPE == 'Fix':
            self.scheduler = None
        elif lr_policy.TYPE == 'Step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size = lr_policy.STEP_SIZE, 
                gamma = lr_policy.GAMMA
            )
        elif lr_policy.TYPE == 'MultiStep':
            self.scheduler = lr_scheduler.create(
                'MultiStep', 
                self.optimizer,
                milestones = lr_policy.STEPS,
                gamma = lr_policy.GAMMA
            )
        elif lr_policy.TYPE == 'Noam':
            self.scheduler = lr_scheduler.create(
                'Noam', 
                self.optimizer,
                model_size = lr_policy.MODEL_SIZE,
                factor = 1.0,
                warmup = warmup_steps,
            )
        elif lr_policy.TYPE == 'WarmStepLR':
            self.scheduler = lr_scheduler.create(
                'WarmStepLR',
                self.optimizer,
                gamma = lr_policy.GAMMA,
                step_size = lr_policy.STEP_SIZE,
                decay_start = lr_policy.STEP_SIZE*2,
            )
        elif lr_policy.TYPE == 'warmup_constant':
            self.scheduler = warmup_lr.WarmupConstantSchedule(
                self.optimizer, 
                warmup_steps=warmup_steps)
        elif lr_policy.TYPE == 'warmup_linear':
            self.scheduler = warmup_lr.WarmupLinearSchedule(
                self.optimizer, 
                min_lr=min_lr/base_lr,
                warmup_steps=warmup_steps,
                t_total=n_steps)
        elif lr_policy.TYPE == 'warmup_cosine':
            self.scheduler = warmup_lr.WarmupCosineSchedule(
                self.optimizer, 
                min_lr=min_lr/base_lr,
                warmup_steps=warmup_steps,
                t_total=n_steps)
        elif lr_policy.TYPE == 'warmup_multistep':
            steps = [step * step_per_epoch for step in lr_policy.STEPS]
            self.scheduler = warmup_lr.WarmupMultiStepLR(
                self.optimizer,
                milestones=steps,
                gamma=lr_policy.GAMMA,
                warmup_factor=0,
                warmup_iters=warmup_steps,
                warmup_method="linear",
            )
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def scheduler_step(self, lrs_type, val=None):
        if self.scheduler is None:
            return

        if lrs_type == self.step_type:
            self.scheduler.step(val)

    def get_lr(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr.append(param_group['lr'])
        lr = sorted(list(set(lr)))
        return lr
