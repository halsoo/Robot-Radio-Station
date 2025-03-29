from pathlib import Path
from omegaconf import DictConfig

import torch


def wandb_style_config_to_omega_config(wandb_conf):
  # remove wandb related config
  for wandb_key in ["wandb_version", "_wandb"]:
    if wandb_key in wandb_conf:
      del wandb_conf[wandb_key] # wandb-related config should not be overrided! 

  # remove nonnecessary fields such as desc and value
  for key in wandb_conf:
    if type(wandb_conf[key]) == DictConfig:
      if 'desc' in wandb_conf[key]:
        del wandb_conf[key]['desc']
      if 'value' in wandb_conf[key]:
        wandb_conf[key] = wandb_conf[key]['value']
  
  return wandb_conf


def evaluate_recommendation(
  model, 
  test_sequence, # batch of 50 songs
  infer_len=5, 
  cluster_top_k=10, 
  track_top_k=10
):
  assert len(test_sequence) == 50, "The test sequence must contain exactly 50 songs."

  condition = torch.tensor([ 
    s[:20] for s in test_sequence 
  ])
  
  ground_truth = [
    set(s[20:]) for s in test_sequence
  ]

  predictions = model.inference(
    condition=condition,
    infer_len=infer_len,
    cluster_top_k=cluster_top_k,
    track_top_k=track_top_k
  )
  
  prediction = predictions.detach().cpu().numpy().tolist()

  total_hit_count = 0
  total_hit_rate = 0.0
  
  for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
    hit_cnt = len(set(pred) & gt)
    hit_rate = hit_cnt / infer_len
    
    total_hit_count += hit_cnt
    total_hit_rate += hit_rate
  
  macro_hit_rate = total_hit_rate / len(predictions)
  micro_hit_rate = total_hit_count / (len(predictions) * infer_len)
  

  return macro_hit_rate, micro_hit_rate