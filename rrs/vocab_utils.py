from typing import Union, List
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    self.tok2idx = { tok: idx for idx, tok in enumerate(self.vocab) }
    self.size = len(self.vocab)
    
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
    with open(vocab_txt_fn, 'r', encoding='utf-8') as f:
      vocab = [ l.rstrip() for l in f.readlines() ]
    
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
    return self.encode(*args, **kwargs)
  
  # encode input string to list of token indices
  def encode(self, words:list) -> List[int]:
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


class ClusterVocab():
  def __init__(
    self, 
    track_to_cluster, 
    cluster_to_tracks
  ):
    self.track_to_cluster = track_to_cluster
    self.cluster_to_tracks = cluster_to_tracks
    
    # Get unique clusters and create index mapping
    self.unique_clusters = list(sorted(list(set(
      self.track_to_cluster.values()
    ))))
    self.unique_clusters = self.unique_clusters
    self.num_clusters = len(self.unique_clusters)
    self.cluster_to_idx = {
      cluster: idx 
      for idx, cluster in enumerate(self.unique_clusters)
    }
    self.idx_to_cluster = {
      idx: cluster 
      for cluster, idx in self.cluster_to_idx.items()
    }
    
    # Create mapping from songs to their position within cluster
    self.track_to_position = {}
    self.cluster_sizes = {}  # Track the size of each cluster
    
    for cluster_id, tracks in self.cluster_to_tracks.items():
      self.cluster_sizes[cluster_id] = len(tracks)
      
      for position, track_uri in enumerate(tracks):
        self.track_to_position[track_uri] = position
    
    # Calculate the maximum position across all clusters
    self.max_position = max(self.cluster_sizes.values()) if self.cluster_sizes else 0
  
  
  def get_cluster_idx(self, track_uri):
    cluster = self.track_to_cluster.get(track_uri, 0)
    return self.cluster_to_idx.get(cluster, 0)
  
  
  def get_position_in_cluster(self, track_uri):
    position = self.track_to_position.get(track_uri, 0)
    return position
  
  
  def add_new_song(self, track_uri, cluster_id, features, reference_feature_dict):
    self.track_to_cluster[track_uri] = cluster_id
    
    if cluster_id in self.cluster_to_tracks:
      self.cluster_to_tracks[cluster_id].append(track_uri)
    else:
      self.cluster_to_tracks[cluster_id] = [track_uri]
    
    
    similar_position = 0
    max_similarity = -1
    
    cluster_tracks = [
      t
      for t in self.cluster_to_tracks.get(cluster_id, []) 
      if t != track_uri and t in reference_feature_dict
    ]
    
    if cluster_tracks and features:
      for existing_song in cluster_tracks:
        if existing_song in reference_feature_dict:
          existing_features = reference_feature_dict[existing_song]
    
          similarity = self._calculate_similarity(features, existing_features)      
          if similarity > max_similarity:
            max_similarity = similarity
            similar_position = self.track_to_position.get(existing_song, 0)
    
    self.track_to_position[track_uri] = similar_position
    
    if cluster_id not in self.unique_clusters:
      self.unique_clusters.append(cluster_id)
      self.unique_clusters.sort()
      self.num_clusters = len(self.unique_clusters)
      self.cluster_to_idx = {
        cluster: idx 
        for idx, cluster in enumerate(self.unique_clusters)
      }
      self.idx_to_cluster = {
        idx: cluster 
        for cluster, idx in self.cluster_to_idx.items()
      }
    
    return similar_position
  
  
  def _calculate_similarity(self, features1, features2):
    common_keys = set(features1.keys()) & set(features2.keys())
    
    if not common_keys:
      return 0.0
    
    vec1 = [features1[k] for k in common_keys]
    vec2 = [features2[k] for k in common_keys]
    
    return cosine_similarity([vec1], [vec2])[0][0]
  
  
  def get_tracks_in_cluster(self, cluster_id):
    """Get all songs in a given cluster."""
    return self.cluster_to_tracks.get(cluster_id, [])
  
  
  def get_cluster_for_song(self, track_uri):
    """Get the cluster ID for a song."""
    return self.track_to_cluster.get(track_uri, 0)
  
  
  def __len__(self):
    """Get the number of unique clusters."""
    return self.num_clusters