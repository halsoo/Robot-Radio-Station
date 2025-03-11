import time
import os
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Subset

import wandb
from tqdm.auto import tqdm

from .model_zoo import NaiveDecoderOnlyRecommender
from .data_utils import PadCollator, InferenceCollator
from .data_utils import SMPInferenceDataset


class DecoderOnlyTrainer:
  def __init__(
    self,
    model,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    loss_fn, 
    train_set, 
    valid_set, 
    save_dir: str, 
    vocab,
    use_fp16:bool,
    batch_size:int,
    config,
    wandb_run=None,
  ):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_set = train_set
    self.valid_set = valid_set
    self.vocab = vocab
    self.batch_size = batch_size
    self.config = config

    self.train_loader = self.generate_data_loader(train_set, shuffle=True, drop_last=True)
    self.valid_loader = self.generate_data_loader(valid_set, shuffle=False, drop_last=False)

    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(exist_ok=True, parents=True)
    
    self.device=config.train_params.device
    self.model.to(self.device)
    
    if use_fp16:
      self.use_fp16 = True
      self.scaler = torch.cuda.amp.GradScaler()
    else:
      self.use_fp16 = False
    
    self.grad_clip=config.train_params.grad_clip
    
    self.num_iter_per_train_log = config.train_params.num_iter_per_train_log
    self.num_iter_per_validation = config.train_params.num_iter_per_validation
    self.num_iter_per_inference = config.train_params.num_iter_per_inference
    self.num_iter_per_checkpoint = config.train_params.num_iter_per_checkpoint
    self.num_inference = config.inference_params.num_inference
    
    self.log=config.general.log
    self.infer=config.general.infer
    self.wandb_run = wandb_run
    
    self.infer_table = None
    if self.wandb_run and self.log and self.infer:
      self.infer_table = wandb.Table(columns=['n_iter', 'condition', 'GT', 'predictions'])
    
    self.best_valid_accuracy = 0
    self.best_valid_loss = 1e9
    
    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []


  def save_model(self, path):
    torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)
  
  
  def generate_data_loader(self, dataset, shuffle=False, drop_last=False) -> DataLoader:
    collate_fn = PadCollator( self.vocab.pad_idx )
    
    return DataLoader(
      dataset,
      batch_size=self.batch_size, 
      shuffle=shuffle, 
      drop_last=drop_last,
      collate_fn=collate_fn,
      num_workers=self.config.general.num_workers,
      prefetch_factor=self.config.general.prefetch_factor
    )


  def train_by_num_iter(self, num_iters):
    generator = iter(self.train_loader)
    
    for i in tqdm(range(num_iters)):
      i += 1 # 1-indexed
      
      try:
        batch = next(generator)
      except StopIteration:
        self.train_loader = self.generate_data_loader(self.train_set, shuffle=True, drop_last=True)
        generator = iter(self.train_loader)
        batch = next(generator)

      self.model.train()

      _, loss_dict = self._train_by_single_batch(batch)
      loss_dict = self._rename_dict(loss_dict, 'train')
      
      if self.log and i % self.num_iter_per_train_log == 0:
        self.wandb_run.log(loss_dict, step=i)
      
      if i % self.num_iter_per_validation == 0:
        self.model.eval()
        validation_loss, validation_acc, validation_metric_dict = self.validate()
        validation_metric_dict['acc'] = validation_acc
        validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
        
        if validation_loss < self.best_valid_loss:
          self.best_valid_loss = validation_loss
          self.save_model(self.save_dir / f'best.pt')
          print(f"Best model saved at {self.save_dir / 'best.pt'}")
        
        if self.log:
          self.wandb_run.log(validation_metric_dict, step=i)
        
      if self.infer and i % self.num_iter_per_inference == 0:
        self.model.eval()
        self.inference_and_log(i)
      
      if i % self.num_iter_per_checkpoint == 0:
        self.save_model(self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt')
        print(f"Checkpoint : {i}th iter : valid loss {validation_loss:.4f} : {self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt'}")
    
    # save last checkpoint
    self.save_model(self.save_dir / f'iter{num_iters}_loss{validation_loss:.4f}.pt')


  def _train_by_single_batch(self, batch):
    start_time = time.time()
    
    loss, _, loss_dict = self._get_loss_pred_from_single_batch(batch)
    
    if self.use_fp16:
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.scaler.step(self.optimizer)
      self.scaler.update()
    
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
    
    self.optimizer.zero_grad()
    
    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.scheduler is not None:
      self.scheduler.step()
    
    loss_dict['time'] = time.time() - start_time
    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    
    return loss.item(), loss_dict
  
  
  def _get_loss_pred_from_single_batch(self, batch):
    seq, tgt, _ = batch
    seq = seq.to(self.device)
    tgt = tgt.to(self.device)
    
    if self.use_fp16:
      with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = self.model(seq)
        loss = self.loss_fn(logits, tgt)

    else:
      logits = self.model(seq)
      loss = self.loss_fn(logits, tgt)
    
    loss_dict = { 'total': loss.item() }
    
    return loss, logits, loss_dict
  
  
  def _get_valid_loss_and_acc_from_batch(self, batch):
    seq, tgt, mask = batch
    loss, logits, loss_dict = self._get_loss_pred_from_single_batch(batch)
    
    num_tokens = torch.sum(mask) 
    tgt = tgt.to(self.device) # N x T
    mask = mask.to(self.device) # N x T

    probs = torch.softmax(logits, dim=-1)
    prob_with_mask = torch.argmax(probs, dim=-1) * mask
    tgt_with_mask = tgt * mask
    num_correct_guess = torch.sum(prob_with_mask == tgt_with_mask) - torch.sum(mask == 0)

    validation_loss = loss.item() * num_tokens
    num_correct_guess = num_correct_guess.item()
    
    return validation_loss, num_tokens, num_correct_guess, loss_dict


  @torch.inference_mode()
  def validate(self):
    loader = self.valid_loader
    total_validation_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    validation_metric_dict = defaultdict(float)
    
    for batch in tqdm(loader, leave=False):
      validation_loss, num_tokens, num_correct_guess, loss_dict = self._get_valid_loss_and_acc_from_batch(batch)
      total_validation_loss += validation_loss
      total_num_tokens += num_tokens
      total_num_correct_guess += num_correct_guess
      
      for key, value in loss_dict.items():
        validation_metric_dict[key] += value * num_tokens
    
    for key in validation_metric_dict.keys():
      validation_metric_dict[key] /= total_num_tokens
    
    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_tokens, validation_metric_dict


  def _rename_dict(self, adict, prefix='train'):
    keys = list(adict.keys())
    for key in keys:
      adict[f'{prefix}.{key}'] = adict.pop(key)
    return dict(adict)


  @torch.inference_mode()
  def inference_and_log(self, n_iter):
    inferenced_table = None
    if self.infer_table:
      inferenced_table = wandb.Table(
        columns=self.infer_table.columns,
        data=self.infer_table.data
      )
    
    # get inference loader for batch inference
    infer_set = SMPInferenceDataset.from_smp_dataset(self.valid_set, self.config.inference_params.infer_length, self.config.inference_params.condition_length)
    infer_set = Subset(infer_set, range(self.num_inference))
    
    collate_fn = InferenceCollator(self.vocab.pad_idx)
    
    infer_loader = DataLoader(
      infer_set,
      batch_size=self.num_inference, 
      shuffle=False, 
      drop_last=True,
      collate_fn=collate_fn,
    )
    
    start_time = time.time()
    
    condition, GT = next(iter(infer_loader))
    condition = condition.to(self.device)
    
    inferenced_output = self.model.inference(
      condition,
      self.config.inference_params.infer_length,
      sampling_method=self.config.inference_params.sampling.method, 
      threshold=self.config.inference_params.sampling.threshold, 
      temperature=self.config.inference_params.sampling.temperature, 
      manual_seed=-1
    )
    
    print(f"Inference: {n_iter}th iter: Time: {time.time() - start_time:.4f}")
    
    for i in range(self.num_inference):
      # decode infered outputs    
      inf_out = inferenced_output[i]
      inf_out = inf_out.to(torch.long)
      inf_out = inf_out.to('cpu')
      decoded = self.vocab.decode(inf_out)
      
      if inferenced_table:
        inferenced_table.add_data(n_iter, condition[i], GT[i], decoded)
    
    if self.log and inferenced_table:
      self.wandb_run.log({"inferenced": inferenced_table})
      self.infer_table = inferenced_table
      
      print(f"Inference: {n_iter}th iter: Log: Done")