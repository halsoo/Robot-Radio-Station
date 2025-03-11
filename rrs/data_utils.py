import random
from collections import defaultdict
from pathlib import Path
from typing import Union, List
import csv
import json

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from . import vocab_utils



class InferenceCollator():
  def __init__(self, token_pad_value):
    self.token_pad_value = token_pad_value
  
  def __call__(self, batch):
    condition, GT = zip(*batch)
    
    condition = torch.stack([ torch.tensor(c, dtype=torch.long) for c in condition ])
    
    max_length = max(len(g) for g in GT)
    padded_GT = []
    for g in GT:
      pad_len = max(0, max_length - len(g))
      padded_GT.append(
        torch.cat([
          torch.tensor(g, dtype=torch.long),
          torch.ones(pad_len, dtype=torch.long) * self.token_pad_value
        ])
      )
    
    return condition, torch.stack(padded_GT)



class PadCollator(object):
  def __init__(self, token_pad_value):
    self.token_pad_value = token_pad_value
  
  
  def __call__(self, batch):
    seq, tgt = zip(*batch)
    
    padded_seq = []
    padded_tgt = []
    token_masks = []
    
    max_length = max(len(s) for s in seq)
    for s, t in zip(seq, tgt):
      pad_len = max(0, max_length - len(s))
      
      padded_seq.append(
        torch.cat([
          torch.tensor(s, dtype=torch.long), 
          torch.ones(pad_len, dtype=torch.long) * self.token_pad_value
        ])
      )
      padded_tgt.append(
        torch.cat([
          torch.tensor(t, dtype=torch.long),
          torch.ones(pad_len, dtype=torch.long) * self.token_pad_value
        ])
      )
      token_masks.append(
        torch.cat([
          torch.ones(len(s), dtype=torch.bool), 
          torch.zeros(pad_len, dtype=torch.bool)
        ])
      )
    
    return torch.stack(padded_seq), torch.stack(padded_tgt), torch.stack(token_masks)



class SMPDataset():
  def __init__(
    self, 
    data_path, # /path/to/dataset/file.pt
    max_length:int,
    vocab:vocab_utils.NaiveTrackVocab,
  ) -> None:
    self.data_path = data_path
    self.max_length = max_length
    self.vocab = vocab
    
    self.data = self._load_data()
  
  
  def _load_data(self) -> torch.Tensor:
    return torch.load(self.data_path)
  
  
  def __len__(self) -> int:
    return len(self.data)
  
  
  def __getitem__(self, idx):
    seq = self.data[idx]
    seq = self.vocab(seq)
    seq = [self.vocab.sos_idx] + seq + [self.vocab.eos_idx]
    
    if len(seq) > self.max_length: # in this case, we sample random subsequence
      seq_len = len(seq)
      start_idx = random.randint(0, seq_len - self.max_length)
      seq = seq[start_idx:start_idx+self.max_length]
      
      if start_idx > 0:
        seq = [self.vocab.sos_idx] + seq
    
    return seq[:-1], seq[1:]



class SMPInferenceDataset(SMPDataset):
  def __init__(
    self, 
    data_path, # /path/to/dataset/file.csv
    max_length:int,
    infer_length,
    condition_length,
    vocab:vocab_utils.NaiveTrackVocab,
  ) -> None:
    super().__init__(data_path, max_length, vocab)
    self.infer_length = infer_length
    self.condition_length = condition_length
  
  
  def __getitem__(self, idx):
    seq = self.data[idx]
    seq = self.vocab(seq)
    condition = [self.vocab.sos_idx] + seq[:self.condition_length]
    GT = seq[self.condition_length:self.condition_length+self.infer_length]
    
    return condition, GT
  
  
  @classmethod
  def from_smp_dataset(cls, dataset, infer_length, condition_length):
    new_set = cls(
      dataset.data_path, 
      dataset.max_length, 
      infer_length, 
      condition_length, 
      dataset.vocab
    )
    
    return new_set



class SMPDatasetMaker(): 
  def __init__(
    self, 
    data_path:Union[str, Path],
    max_length:int,
    vocab_name:str,
    num_special_tokens:int=2,
  ):
    self.data_path = Path(data_path)
    self.max_length = max_length
    self.num_special_tokens = num_special_tokens
    
    self.vocab = getattr(vocab_utils, vocab_name)( 
      vocab_txt_fn = self.data_path / 'vocabulary.txt', 
      num_special_tokens = self.num_special_tokens
    )
  
  
  def get_datasets(self):
    train_dataset = SMPDataset(self.data_path / 'train-segments.pt', self.max_length, self.vocab)
    valid_dataset = SMPDataset(self.data_path / 'valid-segments.pt', self.max_length, self.vocab)
    # test_dataset = SMPDataset(self.data_path / 'test-segments.csv', self.max_length, self.vocab)
    
    return train_dataset, valid_dataset, None