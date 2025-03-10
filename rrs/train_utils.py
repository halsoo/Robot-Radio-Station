import math

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from collections import defaultdict

from .utils import add_conti_for_single_feature


########################### Loss function ################################

class CrossEntropyLoss():
  def __init__(self, ignore_index):
    self.ignore_index = ignore_index
  
  
  def get_nll_loss(self, logits, target):
    """
    logits:
      1. [N, T, vocab_size]
      2. [N*T, vocab_size]
    """
    assert logits.ndim <= 3, f'logits ndim should be leq than 3, logits ndim is {logits.ndim}'

    if logits.ndim == 3: # [N, T, vocab_size]
      logits = logits.flatten(0, 1) # [N*T, vocab_size]
    
    if target.ndim == 2: # [N, T]
      target = target.flatten(0, 1) # [N*T]
    
    loss = F.cross_entropy(logits, target, ignore_index=self.ignore_index, reduction='mean')
    
    return loss
  
  
  def __call__(self, logits, target):
    return self.get_nll_loss(logits, target)



########################### Learning rate Scheduler ################################
'''
This scheduler is from  https://gaussian37.github.io/dl-pytorch-lr_scheduler/#custom-cosineannealingwarmrestarts-1
It's basically a cosine annealing scheduler with warm restarts including two methods, warm up start and reducing maximum lr.
'''

class CosineAnnealingWarmUpRestarts(_LRScheduler):
  def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1, eta_min=0):
    if T_0 <= 0 or not isinstance(T_0, int):
      raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
    if T_mult < 1 or not isinstance(T_mult, int):
      raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
    if T_up < 0 or not isinstance(T_up, int):
      raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
    self.T_0 = T_0
    self.T_mult = T_mult
    self.base_eta_max = eta_max
    self.eta_max = eta_max
    self.T_up = T_up
    self.T_i = T_0
    self.gamma = gamma
    self.cycle = 0
    self.T_cur = last_epoch
    super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
  
  
  def get_lr(self):
    if self.T_cur == -1:
      return self.base_lrs
    elif self.T_cur < self.T_up:
      return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
    else:
      return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
          for base_lr in self.base_lrs]


  def step(self, epoch=None):
    if epoch is None:
      epoch = self.last_epoch + 1
      self.T_cur = self.T_cur + 1
      if self.T_cur >= self.T_i:
        self.cycle += 1
        self.T_cur = self.T_cur - self.T_i
        self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
    else:
      if epoch >= self.T_0:
        if self.T_mult == 1:
          self.T_cur = epoch % self.T_0
          self.cycle = epoch // self.T_0
        else:
          n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
          self.cycle = n
          self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
          self.T_i = self.T_0 * self.T_mult ** (n)
      else:
        self.T_i = self.T_0
        self.T_cur = epoch
        
    self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
    self.last_epoch = math.floor(epoch)
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr



class CosineLRScheduler(_LRScheduler):
  """Cosine LR scheduler.
  Args:
    optimizer (Optimizer): Torch optimizer.
    warmup_steps (int): Number of warmup steps.
    total_steps (int): Total number of steps.
    lr_min_ratio (float): Minimum learning rate.
    cycle_length (float): Cycle length.
  """
  def __init__(
    self, 
    optimizer:Optimizer, 
    total_steps:int, 
    warmup_steps:int,
    lr_min_ratio:float = 0.0, 
    cycle_length:float = 1.0
  ):
    self.warmup_steps = warmup_steps
    assert self.warmup_steps >= 0
    self.total_steps = total_steps
    assert self.total_steps >= 0
    self.lr_min_ratio = lr_min_ratio
    self.cycle_length = cycle_length
    super().__init__(optimizer)


  def _get_sched_lr(self, lr: float, step: int):
    if step < self.warmup_steps:
      lr_ratio = step / self.warmup_steps
      lr = lr_ratio * lr
    elif step <= self.total_steps:
      s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
      lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * \
        (1. + math.cos(math.pi * s / self.cycle_length))
      lr = lr_ratio * lr
    else:
      lr_ratio = self.lr_min_ratio
      lr = lr_ratio * lr
    return lr


  def get_lr(self):
    return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]