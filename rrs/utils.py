from pathlib import Path
from omegaconf import DictConfig


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