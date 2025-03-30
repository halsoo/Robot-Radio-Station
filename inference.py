import os
import argparse
from pathlib import Path
from typing import List, Union

from time import time
from datetime import datetime

import csv
import json

from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

from rrs.vocab_utils import ClusterVocab
from rrs.cluster_utils import get_average_embedding, estimate_new_song_popularity, assign_new_song_to_cluster
from rrs.model_zoo import FeatureClusterRecommender

from gensim.models import Word2Vec


HOME = Path.home()
CWD = Path.cwd()
get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


def load_config(wandb_dir):
  config_file = wandb_dir / 'files' / 'config.yaml'
  config = OmegaConf.load(config_file)

  return config


def prepare_model(config, vocab, pt_path):
  nn_params = config.nn_params
  max_length = config.train_params.max_length
  
  # init model from config
  model = FeatureClusterRecommender(
    dim=nn_params.dim,
    depth=nn_params.depth,
    heads=nn_params.heads,
    dropout=nn_params.dropout,
    vocab=vocab,
    max_length=max_length,
  )
  
  # load pretrained model
  pt = torch.load(pt_path, map_location="cpu", weights_only=False)
  model.load_state_dict(pt["model"])
  
  return model


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
    "--data-csv-dir",
    required=True,
    type=str,
    help="abolute path to the directory containing test data csv",
  )
  parser.add_argument(
    "--checkpoint",
    required=False,
    type=str,
    default='best',
    help="select checkpoint, 'latest', 'best', or iteration number",
  )
  
  return parser


if __name__ == "__main__":
  parser = get_argument_parser()
  args = parser.parse_args()

  wandb_dir = Path.cwd() / 'wandb' / args.experiment
  pt_path = wandb_dir / 'files' / 'checkpoints' / 'best.pt'

  device = torch.device('cpu')
  
  if torch.cuda.is_available():
    print("CUDA is available, using GPU")
    device = torch.device("cuda")
  
  # load config
  config = load_config(wandb_dir)
  
  # # load vocab
  # track_to_cluster = torch.load(CWD / 'data' /'track_to_cluster.pt', weights_only=False)
  # cluster_to_tracks = torch.load(CWD / 'data' /'cluster_to_tracks.pt', weights_only=False)
  
  # vocab = ClusterVocab(
  #   track_to_cluster=track_to_cluster,
  #   cluster_to_tracks=cluster_to_tracks,
  # )
  
  # # Load model
  # print(f"Loading model from {pt_path}")
  # model = prepare_model(config, vocab, pt_path)
  # model.to(device)
  
  # # load encoded data
  # print(f"Loading encoded data...")
  # total_enc = torch.load(CWD / 'test_data' / 'encoded_playlists.pt', weights_only=False, map_location='cpu')
  
  # # Inference and save
  # print(f"Running inference...")
  # pbar = tqdm(total_enc, dynamic_ncols=True)
  # for f_stem, data_enc in pbar:
  #   pbar.set_description(f"processing {f_stem}")
    
  #   infer_len = 5
  #   if f_stem != 'demo_data_p':
  #     infer_len = 20
    
  #   cond = torch.tensor(data_enc, dtype=torch.long).unsqueeze(0)
  #   cond = cond.to(model.device)
    
  #   breakpoint()
    
  #   # output: list length of infer_len
  #   output = model.inference(
  #     cond,
  #     infer_len=infer_len,
  #     cluster_top_k=100,
  #     track_top_k =10,
  #     temperature=1.5,
  #     manual_seed=-1
  #   )
    
  #   # Save conditions
  #   pbar.set_description(f"saving conds: {f_stem}")
  #   with open(CWD / 'test_data' / f'{f_stem}.csv', 'r', encoding='utf-8') as f:
  #     reader = csv.reader(f)
  #     _ = next(reader)
  #     data = list(reader)
    
  #   if f_stem != 'demo_data_p':
  #     data = data[:30]
    
  #   with open(CWD / 'test_data' / f'{f_stem}_conds.txt', 'w', encoding='utf-8') as f:
  #     for d in data:
  #       f.write(d[0]+'\n')
    
  #   # Save output
  #   pbar.set_description(f"saving output: {f_stem}")
  #   with open(CWD / 'test_data' / f'{f_stem}_infs.txt', 'w', encoding='utf-8') as f:
  #     for o in output:
  #       f.write(o+'\n')
  
  # load data
  print(f"Loading necessary data files...")
  start = time()
  
  track_to_artist = torch.load(CWD / 'data' /'track_uri_to_artist.pt', weights_only=False)
  track_to_cluster = torch.load(CWD / 'data' /'track_to_cluster.pt', weights_only=False)
  cluster_to_tracks = torch.load(CWD / 'data' /'cluster_to_tracks.pt', weights_only=False)
  
  vocab = ClusterVocab(
    track_to_cluster=track_to_cluster,
    cluster_to_tracks=cluster_to_tracks,
  )
  
  with open(CWD / 'data' / 'uniq_tracks_w_features.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    _ = next(reader)  # skip header
    uniq_trscks = { row[0]: row for row in reader }
    track_to_popularity = {
      row[0]: {
        'artist_popularity': float(row[-3]),
        'album_popularity': float(row[-2]),
        'track_popularity': float(row[-1]),
      }
      for row in reader
    }
  
  word_to_vector = Word2Vec.load(
    str( CWD / 'models' / 'ml' / 'word2vec_music.model' )
  )
  word_vectors = word_to_vector.wv
  
  audio_feature_view = torch.load(CWD / 'data' / 'audio_feature_view.pt', weights_only=False)
  metadata_feature_view = torch.load(CWD / 'data' / 'metadata_feature_view.pt', weights_only=False)
  
  view_models = {}
  view_clusters = {}
  audio_clusters = torch.load(CWD / 'data' / 'audio_clusters.pt', weights_only=False)
  view_models['audio'] = {
    'model': audio_clusters['cluster_model'],
    'scaler': audio_clusters['scaler'],
    'feature_keys': audio_clusters['feature_keys'],
  }
  view_clusters['audio'] = audio_clusters['clusters']
  metadata_clusters = torch.load(CWD / 'data' / 'metadata_clusters.pt', weights_only=False)
  view_models['metadata'] = {
    'model': metadata_clusters['cluster_model'],
    'scaler': metadata_clusters['scaler'],
    'feature_keys': metadata_clusters['feature_keys'],
  }
  view_clusters['metadata'] = metadata_clusters['clusters']
  
  print("Necessary data files loaded in {:.2f} seconds".format(time() - start))
  # track_uri,artist,album,track,yt_id,yt_title,duration_seconds,score,acousticness,danceability,speechiness,instrumentalness,key,liveness,loudness,mode,tempo,valence,artist_popularity,album_popularity,track_popularity
  
  # Load model
  print(f"Loading model from {pt_path}")
  model = prepare_model(config, vocab, pt_path)
  model.to(device)
  
  print(f"Loading data from {args.data_csv_dir}")
  csv_files = list(sorted(Path(args.data_csv_dir).glob('*_p.csv')))
  print(f"Found {len(csv_files)} files")

  total_enc = []
  pbar = tqdm(csv_files, dynamic_ncols=True)
  for csv_file in pbar:
    pbar.set_description(f"processing {csv_file.name}")
    with open(csv_file, 'r', encoding='utf-8') as f:
      reader = csv.reader(f)
      _ = next(reader)  # skip header
      data = list(reader)

    infer_len = 5
    
    if csv_file.stem != 'demo_data_p':
      data = data[:30]
      infer_len = 20
    
    data_enc = []
    for r in tqdm(data, desc='Encoding', dynamic_ncols=True, leave=False):
      if r[0] not in track_to_cluster:
        new_pop = estimate_new_song_popularity(
          { 'artist': r[1], 'album': r[2], 'track': r[3] },
          track_to_artist,
          track_to_popularity,
        )
        cluster_idx, position_in_cluster = assign_new_song_to_cluster(
          r[0],
          {
            'acousticness': float(r[8]), 
            'danceability': float(r[9]), 
            'speechiness': float(r[10]), 
            'instrumentalness': float(r[11]), 
            'key': float(r[12]), 
            'liveness': float(r[13]), 
            'loudness': float(r[14]), 
            'mode': float(r[15]), 
            'tempo': float(r[16]), 
            'valence': float(r[17]), 
          },
          {
            **new_pop,
            'artist': get_average_embedding(r[1], word_vectors),
            'album': get_average_embedding(r[2], word_vectors),
            'track': get_average_embedding(r[3], word_vectors),
            'duration': float(r[6]),
          },
          audio_feature_view,
          metadata_feature_view,
          view_models,
          view_clusters,
          vocab
        )
      else:
        track_uri = r[0]
        cluster_idx = vocab.get_cluster_idx(track_uri)
        position_in_cluster = vocab.get_position_in_cluster(track_uri)
      
      data_enc.append([
        cluster_idx, position_in_cluster
      ])
    
    pbar.set_description(f"Inferencing...")
    cond = torch.tensor(data_enc, dtype=torch.long).unsqueeze(0)
    cond = cond.to(model.device)
    
    # output: list length of infer_len
    output = model.inference(
      cond,
      infer_len=infer_len,
      cluster_top_k=100,
      track_top_k =10,
      temperature=1.5,
      manual_seed=-1
    )
    
    # Save conditions
    pbar.set_description(f"saving conds: {csv_file.stem}")
    with open(CWD / 'test_data' / f'{csv_file.stem}_conds.txt', 'w', encoding='utf-8') as f:
      for d in data:
        f.write(d[0]+'\n')
    
    # Save output
    pbar.set_description(f"saving output: {csv_file.stem}")
    with open(CWD / 'test_data' / f'{csv_file.stem}_infs.txt', 'w', encoding='utf-8') as f:
      for o in output:
          f.write(o+'\n')