from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from x_transformers.x_transformers import LayerIntermediates, AbsolutePositionalEmbedding
from x_transformers import Encoder as TransformerEncoder

from .nn import DecoderOnly
from .sampling_utils import sample


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
    
    self.input_embedder = nn.Embedding(vocab.size, dim)
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


  def forward(self, seq:torch.Tensor):
    emb = self.input_embedder(seq)
    emb += self.pos_enc(emb)
    emb = self.emb_dropout(emb)
    
    hidden = self.decoder(emb)
    hidden = self.decoder_norm(hidden)
    
    logits = self.proj(hidden)
    
    return logits
  
  
  def _run_one_step(self, seq, cache=None):
    emb = self.input_embedder(seq) 
    emb += self.pos_enc(seq)
    emb = self.emb_dropout(emb)
    
    hidden, cache = self.decoder(emb, cache=cache) # B x T x d_model
    hidden = self.decoder_norm(hidden)
    logits = self.proj(hidden)
    
    return logits, cache
  
  
  def _sample_and_update(
    self, logits, total_out, 
    sampling_method=None, threshold=None, temperature=1
  ):
    sampled = sample(logits, sampling_method, threshold, temperature)
    sampled = sampled[:, -1:]
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



class FeatureClusterRecommender(nn.Module):
  """
  2-stage recommender model using encoder-only transformer architecture.
    1. predict next cluster
    2. select song within predicted cluster
  """
  
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
    
    self.cluster_embedding = nn.Embedding(self.vocab.num_clusters, dim//2)
    self.position_in_cluster_embedding = nn.Embedding(self.vocab.max_positions, dim//2)
    self.pos_emb = AbsolutePositionalEmbedding(dim, self.max_length)
    self.emb_dropout = nn.Dropout(dropout)
    
    self.encoder = TransformerEncoder(
      dim=dim,
      depth=depth,
      heads=heads,
      attn_dropout=dropout,
      ff_dropout=dropout,
      attn_flash=True,
    )
    self.encoder_norm = nn.LayerNorm(dim)
    self.encoder_dropout = nn.Dropout(dropout)
    
    # cluster prediction
    self.cluster_predictor = nn.Sequential(
      nn.Linear(dim, dim),
      nn.LayerNorm(dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim, self.num_clusters)
    )
    
    # in-cluster prediction
    self.track_predictor = nn.Sequential(
      nn.Linear(dim, dim),
      nn.LayerNorm(dim),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(dim, dim)
    )


  @property
  def device(self):
    return next(self.parameters()).device


  def forward(self, seq:torch.Tensor):
    # seq: (N, T, 2)
    # last dim: [cluster_idx, position_idx_in_cluster]
    # last k token is masked
    cluster_emb = self.cluster_embedding(seq[:, :, 0]) # N, T, dim//2
    track_pos_emb = self.position_in_cluster_embedding(seq[:, :, 1]) # N, T, dim//2
    emb = torch.cat([cluster_emb, track_pos_emb], dim=-1) # N, T, d
    emb += self.pos_emb(emb)
    
    # before decoder dropout
    emb = self.emb_dropout(emb)
    
    hidden = self.encoder(emb)
    hidden = self.encoder_norm(hidden)
    
    # last hidden state as context
    context = hidden[:, -1, :] # N, d
    
    cluster_logit = self.cluster_predictor(context) # N, num_clusters
    track_embedding = self.track_predictor(context) # N, d
    
    return cluster_logit, track_embedding
  
  
  @torch.inference_mode()
  def inference(
    self,
    condition:list,
    infer_len:int,
    cluster_top_k:int,
    track_top_k:int,
    temperature=1,
    manual_seed=-1
  ):
    encoded_condition = [
      torch.tensor(
        [
          self.vocab.cluster_to_idx[
            self.vocab.track_to_cluster[track_uri]
          ],
          self.track_to_position[track_uri]
        ], 
        dtype=torch.long, device=self.device
      ) 
      for track_uri in condition
    ]
    
    with torch.no_grad():
      next_cluster_logits, next_track_embedding = self.forward(encoded_condition)
    
    next_cluster_probs = F.softmax(next_cluster_logits / temperature, dim=-1)
    top_cluster_values, top_cluster_indices = torch.topk(
      next_cluster_probs, 
      k=min(self.num_clusters, cluster_top_k)
    )
    
    top_clusters = [
      self.unique_clusters[idx.item()] 
      for idx in top_cluster_indices[0]
    ]
    top_probs = [
      val.item() 
      for val in top_cluster_values[0]
    ]
    
    # For each top cluster, find most similar songs
    recommendations = []
    
    for cluster_id, prob in zip(top_clusters, top_probs):
      # get tracks in the cluster
      if cluster_id in self.vocab.cluster_to_tracks:
        cluster_tracks = self.cluster_to_songs[cluster_id]
      else:
        continue
      
      if not cluster_tracks:
        continue
      
      track_indices= [ self.vocab.idx_to_cluster[t_uri] for t_uri in cluster_tracks ]
      track_indices = torch.tensor(track_indices, dtype=torch.long, device=self.device)
      cluster_track_embeddings = self.track_embedding(track_indices)
      
      similarities = F.cosine_similarity(
        next_track_embedding.unsqueeze(1), # 1, 1, d
        cluster_track_embeddings.unsqueeze(0), # 1, ?, d
        dim=2
      )[0]
      
      weighted_similarities = similarities * prob
      
      num_to_get = min(track_top_k, len(cluster_songs))
      if num_to_get > 0:
        top_track_values, top_track_indices = torch.topk(weighted_similarities, k=num_to_get)
        
        for idx, val in zip(top_track_indices, top_track_values):
          track_uri = cluster_songs[idx.item()]
          score = val.item()
          recommendations.append((track_uri, score))
    
    # Sort by score and remove duplicates and songs already in input
    recommendations.sort(key=lambda x: x[1], reverse=True)
    seen = set(condition)
    unique_recommendations = []
    
    for song_id, _ in recommendations:
      if song_id not in seen:
        unique_recommendations.append(song_id)
        seen.add(song_id)
        
        if len(unique_recommendations) >= infer_len:
          break
    
    return unique_recommendations