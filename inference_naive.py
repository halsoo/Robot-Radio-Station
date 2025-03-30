import os
import argparse
from pathlib import Path
from typing import List, Union

from datetime import datetime

import csv
import json

from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from rrs.vocab_utils import NaiveTrackVocab
from rrs.model_zoo import NaiveDecoderOnlyRecommender
from rrs.utils import wandb_style_config_to_omega_config
from rrs.data_utils import InferenceCollator, SMPInferenceDataset


get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


def load_config(wandb_dir):
  timestamp = wandb_dir.stem.split(':')[0].split('-')
  sub_dir = '-'.join(timestamp[:3])
  sub_sub_dir = '-'.join(timestamp[3:])
  
  config_file = Path.cwd() / 'outputs' / sub_dir / sub_sub_dir / '.hydra' / 'config.yaml'
  config = OmegaConf.load(config_file)

  return config


def get_dataset(config, pt_path:str, vocab, infer_length, condition_length)->SMPInferenceDataset:
  return SMPInferenceDataset(
    data_path=Path(pt_path),
    max_length=config.train_params.max_length,
    infer_length=infer_length,
    condition_length=condition_length,
    vocab=vocab
  )


def get_vocab(config):
  vocab_path = Path(config.data.data_path) / 'vocabulary.txt'
  vocab = NaiveTrackVocab(
    vocab_txt_fn=vocab_path,
    num_special_tokens=config.data.vocab.num_special_tokens,
  )

  return vocab


def generate_data_loader(config, dataset, vocab):
  collate_fn = InferenceCollator(vocab.pad_idx)
  batch_size = config.train_params.batch_size
  
  return DataLoader(
    dataset,
    batch_size=batch_size, 
    shuffle=False, 
    drop_last=False,
    collate_fn=collate_fn,
  )


def parse_checkpoint_arg(wandb_dir:Path, checkpoint:Union[str|int]='latest'): # => (Path, int)
  glob_checkpoints = lambda ptr: (wandb_dir / 'files' / 'checkpoints').glob(ptr)

  if checkpoint == 'latest':
    # load latest checkpoint
    pt_path = glob_checkpoints('*.pt')
    pt_path = sorted(pt_path, key=lambda x:int(x.stem.split('_')[0].replace('iter', '')))
    pt_path = list(pt_path)[-1] # largest iteration
    num_iter = int(pt_path.stem.split('_')[0].replace('iter', ''))
  
  elif checkpoint == 'best':
    pt_path = wandb_dir / 'files' / 'checkpoints' / 'best.pt'
    num_iter = -1
  
  else:
    checkpoint = int(checkpoint)
    pt_path = glob_checkpoints(f'iter{checkpoint}_loss*.pt')
    pt_path = list(pt_path)
    
    if len(pt_path) < 0 or len(pt_path) > 1:
      raise ValueError(f"Checkpoint iter{checkpoint} not found")
    
    pt_path = pt_path[0]
    num_iter = int(pt_path.stem.split('_')[0].replace('iter', ''))
  
  return pt_path, num_iter


def prepare_model(config, vocab, pt_path):
  nn_params = config.nn_params
  max_length = config.train_params.max_length
  
  # init model from config
  model = NaiveDecoderOnlyRecommender(
    dim=nn_params.dim,
    depth=nn_params.depth,
    heads=nn_params.heads,
    dropout=nn_params.dropout,
    vocab=vocab,
    max_length=max_length,
  )
  
  # load pretrained model
  pt = torch.load(pt_path, map_location="cpu")
  model.load_state_dict(pt["model"])
  
  return model


def inference(args, wandb_dir, model_pt_path, n_iter):
  device = torch.device('cpu')
  
  if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = torch.device("cuda")
  
  # load config
  config = load_config(wandb_dir)
  
  # load data
  print(f"Loading dataset")
  vocab = get_vocab(config)
  dataset = get_dataset(
    config, 
    args.data_pt_path,
    vocab, 
    config.inference_params.infer_length, 
    config.inference_params.condition_length
  )
  
  data_loader = generate_data_loader(config, dataset, vocab)
  
  # load model
  model = prepare_model(config, vocab, model_pt_path)
  print(f"Loaded {n_iter} model from iteration {n_iter}")
  model.eval()
  model.to(device)
  
  # inference
  conditions = []
  GTs = []
  outputs = []
  
  for batch in tqdm(data_loader):
    condition, GT = batch
    conditions += condition.tolist()
    GTs += GT.tolist()
    
    condition = condition.to(device)
    output = model.inference(
      condition,
      config.inference_params.infer_length,
      sampling_method=config.inference_params.sampling.method, 
      threshold=config.inference_params.sampling.threshold, 
      temperature=config.inference_params.sampling.temperature, 
      manual_seed=-1
    )
    
    output = output.to(device="cpu", dtype=torch.long)
    output = output[:, config.inference_params.condition_length+1:]
    outputs += output.tolist()
  
  # postprocess
  conditions_dec = [ vocab.decode(condition) for condition in conditions ]
  GTs_dec = [ vocab.decode(gt) for gt in GTs ]
  outputs_dec = [ vocab.decode(pred) for pred in outputs ]
  
  total_out_enc = list(zip(conditions, GTs, outputs))
  total_out_dec = list(zip(conditions_dec, GTs_dec, outputs_dec))
  
  return total_out_enc, total_out_dec


def get_argument_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-w",
    "--experiment",
    required=True,
    type=str,
    help="wandb experiment path",
  )
  parser.add_argument(
    "-d",
    "--data-pt-path",
    required=True,
    type=str,
    help="abolute path to the directory containing .pt file",
  )
  parser.add_argument(
    "--checkpoint",
    required=False,
    type=str,
    default='best',
    help="select checkpoint, 'latest', 'best', or iteration number",
  )
  # parser.add_argument(
  #   "--max-length",
  #   required=False,
  #   type=int,
  #   default=700,
  #   help="max length of the inference sequence",
  # )
  
  return parser


if __name__ == "__main__":
  parser = get_argument_parser()
  args = parser.parse_args()

  wandb_dir = Path.cwd() / 'wandb' / args.experiment

  pt_path, n_iter = parse_checkpoint_arg(wandb_dir, args.checkpoint)

  # create output directory if not exists
  output_dir = Path(args.data_pt_path).parent / 'inference_output'
  if not output_dir.exists():
    output_dir.mkdir(parents=False)

  total_enc, total_dec = inference(args, wandb_dir, pt_path, n_iter)
  
  print('saving outputs...')
  
  output_enc_path = output_dir / f'inference_{args.experiment}_iter{args.checkpoint}_enc.json'
  with open(output_enc_path, 'w', encoding='utf-8') as f:
    data = dict(
      data=[ { 'condition': cond, 'GT': gt, 'prediction': pred } for cond, gt, pred in total_enc ],
    )
    json.dump(data, f, indent=2)
  
  output_dec_path = output_dir / f'inference_{args.experiment}_iter{args.checkpoint}_dec.json'
  with open(output_dec_path, 'w', encoding='utf-8') as f:
    data = dict(
      data=[ { 'condition': cond, 'GT': gt, 'prediction': pred } for cond, gt, pred in total_dec ],
    )
    json.dump(data, f, indent=2)