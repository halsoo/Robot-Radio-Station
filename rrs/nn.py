from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from x_transformers import Decoder


class DecoderOnly(nn.Module):
  def __init__(
    self, 
    dim,
    depth,
    heads,
    dropout,
  ):
    super().__init__()
    self.decoder = Decoder(
      dim = dim,
      depth = depth,
      heads = heads,
      attn_dropout = dropout,
      ff_dropout = dropout,
      attn_flash = True,
      cross_attend = False,
    )

  def forward(self, seq, cache=None):
    # inference mode
    if cache is not None:
      if cache.hiddens is None: 
        cache = None
      
      hidden_vec, intermediates = self.decoder(seq, cache=cache, return_hiddens=True)
      
      return hidden_vec, intermediates
    
    # training mode
    return self.decoder(seq)