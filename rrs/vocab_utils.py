from typing import Union, List
from pathlib import Path

import torch


class NaiveTrackVocab():
  def __init__(
    self, 
    vocab_txt_fn:Union[Path,str]=None, 
    num_special_tokens:int=2,
  ) -> None:
    """
    num_special_tokens:
      if 3: pad != sos != eos
      if 2: pad != sos == eos (default)
      if 1: pad == sos == eos
    """
    assert Path(vocab_txt_fn).exists(), 'vocab_txt_fn does not exist.'
    assert num_special_tokens in {1, 2, 3}, 'num_special_tokens should be 1, 2, or 3.'
    
    self.num_special_tokens = num_special_tokens
    self.vocab_txt_fn = vocab_txt_fn
    
    self.vocab = self._load_vocab(vocab_txt_fn)
    self.tok2idx = { tok: idx for idx, tok in enumerate(self.vocabs) }
    self.size = len(self.vocabs)
    
    self.pad_idx, self.sos_idx, self.eos_idx = [ 
      self.tok2idx[t] 
      for t in [self.pad_token, self.sos_token, self.eos_token] 
    ]

  
  def _get_special_tokens(self) -> List[str]:
    match self.num_special_tokens:
      case 3:
        special_tokens = ['<pad>', '<sos>', '<eos>']
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
      case 2:
        special_tokens = ['<pad>', '<sos>']
        self.pad_token = '<pad>'
        self.sos_token = self.eos_token = '<sos>'
      case 1:
        special_tokens = ['<pad>']
        self.pad_token = self.sos_token = self.eos_token = '<pad>'
    
    return special_tokens
  
  
  def _load_vocab(self, vocab_txt_fn:Union[Path,str]) -> List[str]:
    with open(vocab_txt_fn, 'r') as f:
      vocab = [ l.strip() for l in f.readlines() if l.strip() != '' ]
    
    assert (
      len(vocab) == len(set(vocab)),
      'There are duplicated tokens in vocab file.'
    )
    
    num_special_tokens_from_txt = len(set(vocab[:3]) & {'<pad>', '<sos>', '<eos>'})
    
    special_tokens = self._get_special_tokens()
    
    if num_special_tokens_from_txt < 1:
      print('WARNING: There is no special token in vocab file, could be a major issue.')
      vocab = special_tokens + vocab
    
    elif self.num_special_tokens != num_special_tokens_from_txt:
      raise(ValueError, 'Number of special tokens in vocab file is not matched with propvided num_special_tokens.')
    
    return vocab
  
  
  def __call__(self, *args, **kwargs):
    self.encode(*args, **kwargs)
  
  # encode input string to list of token indices
  def encode(self, seq) -> List[int]:
    # split lmx string by space
    words = (
      seq
        .replace('\n', '')
        .split(' ')
    )
    
    # encode words to token indices
    encoded = [ self.tok2idx[w] for w in words ]
    
    return encoded
  
  
  def _get_special_indices(self) -> list[int]:
    return [ self.pad_idx, self.sos_idx, self.eos_idx ]
  
  
  def decode(self, indices:Union[torch.Tensor, List[int]]) -> str:
    if isinstance(indices, torch.Tensor):
      if indices.ndim == 2: # [1, seq_len]
        indices = indices.squeeze(0) # [seq_len]
      
      indices = indices.tolist()
    
    # drop <sos> token
    indices = indices[1:]
    
    # slice indices before first eos token
    if self.eos_idx in indices:
      indices = indices[:indices.index(self.eos_idx)]
    
    special_indices = self._get_special_indices()
    special_indices = set(special_indices)
    
    indices_decoded = [
      self.vocab[idx] for idx in indices 
      if idx not in special_indices # pad, sos, eos
    ]
    
    decoded = ' '.join(indices_decoded)
    
    return decoded