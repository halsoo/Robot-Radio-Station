from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from x_transformers.x_transformers import LayerIntermediates, AbsolutePositionalEmbedding

from .nn import DecoderOnly
from .sample_utils import sample


class AutoRegressiveWrapper(nn.Module):
  def __init__(
    self, 
    *, 
    dim,  
    depth, 
    heads, 
    dropout,
    vocab, 
    max_length,
  ):
    super().__init__()
    self.vocab = vocab
    self.max_length = max_length
    
    self.input_embbedder = nn.Embedding(vocab.size, dim)
    self.emb_dropout = nn.Dropout(dropout)
    self.pos_enc = AbsolutePositionalEmbedding(dim, self.max_length)
    
    self.decoder = DecoderOnly(
      dim=dim,
      depth=depth,
      heads=heads,
      dropout=dropout,
    )
    self.decoder_norm = nn.LayerNorm(dim)
    
    self.proj = nn.Linear(dim, vocab.size)


  @property
  def device(self):
    return next(self.parameters()).device


  def forward(self, seq:torch.Tensor, tgt:torch.Tensor):
    emb = self.input_embedder(seq)
    emb = self.main_tf_dropout(emb)
    emb = emb + self.pos_enc(emb)
    emb = self.emb_dropout(emb)
    
    hidden = self.main_decoder(emb)
    hidden = self.main_norm(hidden)
    
    logits = self.proj(hidden)
    
    return logits
  
  
  def _run_one_step(self, seq, cache=None):
    emb = self.input_embedder(seq) 
    emb += self.pos_enc(seq)
    emb = self.emb_dropout(emb)
    
    hidden, cache = self.decoder(emb, cache=cache) # B x T x d_model
    hidden = self.norm(hidden)
    logits = self.proj(hidden)
    
    return logits, cache
  
  
  def _sample_and_update(
    self, logits, total_out, 
    sampling_method=None, threshold=None, temperature=1
  ):
    
    if logits.ndim == 1:
      logits = logits.unsqueeze(0) # add batch dim
    
    sampled = sample(logits, sampling_method, threshold, temperature)
    total_out = torch.cat([total_out, sampled], dim=-1)
    
    return total_out, sampled
  
  
  @torch.inference_mode()
  def inference(
    self, 
    condition,
    infer_length,
    sampling_method=None, 
    threshold=None, 
    temperature=1, 
    manual_seed=-1
  ):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
    
    total_out = condition
    
    logits, cache = self._run_one_step(
      total_out, 
      cache=LayerIntermediates()
    )
    
    total_out, _ = self._sample_and_update(
      logits, total_out,
      sampling_method=sampling_method,
      threshold=threshold,
      temperature=temperature
    )
    
    for _ in range(infer_length-1):
      logits, cache = self._run_one_step(total_out, cache=cache)
      
      total_out, _ = self._sample_and_update(
        logits, total_out,
        sampling_method=sampling_method, threshold=threshold, temperature=temperature
      )
    
    return total_out



class NaiveDecoderOnlyRecommender(nn.Module):
  def __init__(
    self,
    *,
    dim,
    depth,
    heads,
    dropout,
    vocab,
    max_length,
  ):
    super().__init__()
    
    self.decoder = AutoRegressiveWrapper(
      dim=dim,
      heads=heads,
      depth=depth,
      dropout=dropout,
      vocab=vocab,
      max_length=max_length,
    )


  def forward(self, seq):
    return self.decoder(seq)


  @torch.inference_mode()
  def inference(
    self, 
    condition, infer_length, 
    sampling_method=None, threshold=None, temperature=1, 
    manual_seed=-1
  ):
    total_out = self.decoder.inference(
      condition, infer_length,
      sampling_method=sampling_method, threshold=threshold, temperature=temperature, 
      manual_seed=manual_seed
    )
    
    return total_out