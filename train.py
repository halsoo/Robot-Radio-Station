import os
import copy
from pathlib import Path
from datetime import datetime

import torch

import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from rrs import trainer
from rrs import data_utils
from rrs import model_zoo
from rrs import train_utils


get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def generate_experiment_name(config):
  # add base hyperparameters to the experiment name
  dataset_name = config.data.metadata_dir
  
  net = config.nn_params.net
  dim = config.nn_params.model_dim
  depth = config.nn_params.depth
  dropout = config.nn_params.dropout

  batch_size = config.train_params.batch_size
  lr_decay_rate = config.train_params.decay_step_rate

  # Combine the information into a single string for the experiment name
  # eg. "2024-07-30-21-19-18:example:sqdac:4:stack:CrossAttention_Strategy:dim-128:nLayers-2_6:dropout-0.1:batchSize-8:lrDcay-0.8"
  experiment_name = f"{get_ts()}:{net}:dim-{dim}:nLayers-{depth}:dropout-{dropout}:batchSize-{batch_size}:lrDecay-{lr_decay_rate}:{dataset_name}"
  
  return experiment_name


def setup_log(config):
  experiment_name = generate_experiment_name(config)
  
  if config.general.log:
    wandb_config = config.wandb_config
    
    wandb_run = wandb.init(
      project=wandb_config.project,
      entity=wandb_config.entity,
      name=experiment_name,
      config = OmegaConf.to_container(config)
    )
    
    save_dir = wandb_run.dir + '/checkpoints/'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    (Path('./wandb') / experiment_name).symlink_to(Path(wandb_run.dir).parent, target_is_directory=True)
    
  else:
    wandb_run = None
    save_dir = f'wandb/debug/{experiment_name}/files/checkpoints/'
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config=config, f=str(Path(save_dir).parent / 'config.yaml'))
  
  return wandb_run, save_dir


def prepare_trainer(config, wandb_run, save_dir):
  nn_params = config.nn_params
  data_config = config.data

  max_length = config.train_params.max_length
  
  dataset = getattr(data_utils, data_config.dataset)(
    data_path=data_config.data_path,
    max_length=config.max_length,
    vocab_name=data_config.vocab.name,
    num_special_tokens=data_config.vocab.num_special_tokens,
  )
  
  trainset, validset, _ = dataset.get_datasets()
  
  model = getattr(model_zoo, nn_params.net)(
    dim=nn_params.dim,
    depth=nn_params.depth,
    heads=nn_params.heads,
    dropout=nn_params.dropout,
    vocab=dataset.vocab,
    max_length=max_length,
  )
  
  total_params = sum(p.numel() for p in model.parameters())
  print(f"Total Num Params: {total_params}")
  
  # log in wandb
  if config.general.log:
    wandb_run.log({'nn_total_params': total_params})
  
  # get loss function
  loss_fn = train_utils.CrossEntropyLoss(dataset.vocab.pad_idx)
  
  # get optimizer and scheduler
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.train_params.initial_lr, betas=(0.9, 0.95), eps=1e-08, weight_decay=0.01)
  
  scheduler_dict = {
    'not-using': None,
    'cosineannealingwarmuprestarts': train_utils.CosineAnnealingWarmUpRestarts, 
    'cosinelr': train_utils.CosineLRScheduler,
  }
  
  if scheduler_dict[config.train_params.scheduler] == train_utils.CosineAnnealingWarmUpRestarts:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, T_0=config.train_params.num_steps_per_cycle, T_mult=2, eta_min=0, eta_max=config.train_params.max_lr,  T_up=config.train_params.warmup_steps , gamma=config.train_params.gamma)
  
  elif scheduler_dict[config.train_params.scheduler] == train_utils.CosineLRScheduler:
    scheduler = scheduler_dict[config.train_params.scheduler](optimizer, total_steps=config.train_params.num_iter * config.train_params.decay_step_rate, warmup_steps=config.train_params.warmup_steps, lr_min_ratio=0.1, cycle_length=1.0)
  
  else:
    scheduler = None
  
  return trainer.DecoderOnlyTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    train_set=trainset,
    valid_set=validset,
    save_dir=save_dir,
    vocab=dataset.vocab,
    use_fp16=config.use_fp16,
    batch_size=config.train_params.batch_size,
    config=config,
    wandb_run=wandb_run
  )


def run_train_exp(config):
  wandb_run, save_dir = setup_log(config)
  trainer = prepare_trainer(config, wandb_run, save_dir)
  trainer.train_by_num_iter(config.train_params.num_iter)


@hydra.main(version_base=None, config_path="./config/", config_name="config")
def main(config: DictConfig):
    run_train_exp(config) # single gpu


if __name__ == "__main__":
  main()