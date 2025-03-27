import os
from time import sleep
from datetime import datetime
import subprocess
import tempfile
from pathlib import Path

import math
import random

import csv

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import wandb

import essentia.standard as es
from yt_dlp import YoutubeDL

get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

HOME = Path.home()
CWD = Path.cwd()
TMP = CWD / 'tmp'

VPN_CHANGE_INTERVAL = 20
START_IDX = 0
# CSV_FILE = CWD / 'data' / 'track_yt_infos_2025-03-25-01:43:12.csv' # for rrs-es process
CSV_FILE = CWD / 'data' / 'track_yt_infos_2025-03-25-14:03:19.csv' # for rrs process


def change_location(city_list, city_iter):
  while True: # to avoid iteration error and connection error
    try: 
      city = next(city_iter)
    except StopIteration:
      city_list = random.sample(city_list, len(city_list))
      city_iter = iter(city_list)
      city = next(city_iter)

    cmd = ['nordvpn', 'connect', city]
    try:
      subprocess.run(cmd, check=True)
      return city, city_iter
    except:
      continue


# extract spotify-like audio features with essentia.
def extract_audio_features(audio_file):
  extractor = es.MusicExtractor(lowlevelStats=['mean', 'stdev'])
  features, _ = extractor(audio_file)

  spotify_features = {}
  
  # acousticness [0, 1]
  rolloff = min(features['lowlevel.spectral_rolloff.mean'] / 20000.0, 1.0)
  spotify_features["acousticness"] = 1.0 - rolloff
  
  # danceability [0, 1]
  if features.containsKey('rhythm.danceability'):
    spotify_features["danceability"] = min(max(features['rhythm.danceability'], 0.0), 1.0)
  else: # fallback
    beat_intervals = np.diff(features['rhythm.beats_position']) if features.containsKey('rhythm.beats_position') else [0.5]
    if len(beat_intervals) > 1:
      beat_regularity = 1.0 - min(np.std(beat_intervals) / np.mean(beat_intervals), 1.0)
      spotify_features["danceability"] = min(max(beat_regularity, 0.0), 1.0)
    else:
      spotify_features["danceability"] = 0.5  # Default value
  
  # energy [0, 1]
  if features.containsKey('lowlevel.average_loudness'):
    normalized_loudness = (features['lowlevel.average_loudness'] + 60) / 60
    spotify_features["energy"] = min(max(normalized_loudness, 0.0), 1.0)
  else: # fallback
    spotify_features["energy"] = min(max(features['lowlevel.dynamic_complexity'], 0.0), 1.0)
  
  # speechiness [0, 1]
  # instrumentalness [0, 1]
  if features.containsKey('highlevel.voice_instrumental') and 'vocal' in features['highlevel.voice_instrumental']:
    speech_prob = features['highlevel.voice_instrumental']['vocal']
    speech_prob = min(max(speech_prob, 0.0), 1.0)
    instumental_prob = 1.0 - speech_prob
    spotify_features["speechiness"] = speech_prob
    spotify_features["instrumentalness"] = instumental_prob
  else: # fallback: mfcc
    mfcc_speech_indicator = min(max(features['lowlevel.mfcc.mean'][1] / 100.0, 0.0), 1.0)
    spotify_features["speechiness"] = mfcc_speech_indicator
    spotify_features["instrumentalness"] = 1.0 - mfcc_speech_indicator
  
  # key [0-11] or -1
  key_mapping = { 'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11 }
  detected_key = features['tonal.key_edma.key']
  if detected_key in key_mapping:
    spotify_features["key"] = key_mapping[detected_key]
  else: # fallback: no key detected
    spotify_features["key"] = -1  
  
  # liveness [0, 1]
  if features.containsKey('lowlevel.spectral_flux.mean'):
    normalized_flux = min(features['lowlevel.spectral_flux.mean'] / 0.5, 1.0)
    spotify_features["liveness"] = normalized_flux
  else: # fallback: spetral_centroid
    centroid_stdev = features['lowlevel.spectral_centroid.stdev'] if features.containsKey('lowlevel.spectral_centroid.stdev') else 2000.0
    spotify_features["liveness"] = min(centroid_stdev / 5000.0, 1.0)
  
  # loudness [dB]
  if features.containsKey('lowlevel.loudness_ebu128.integrated'):
    spotify_features["loudness"] = features['lowlevel.loudness_ebu128.integrated']
  else:
    spotify_features["loudness"] = features['lowlevel.average_loudness'] if features.containsKey('lowlevel.average_loudness') else -20.0
  
  # mode: major: 1 / minor: 0
  detected_scale = features['tonal.key_edma.scale']
  spotify_features["mode"] = 1 if detected_scale == "major" else 0
  
  # tempo [BPM]: Direct mapping
  spotify_features["tempo"] = features['rhythm.bpm'] if features.containsKey('rhythm.bpm') else 120.0
  
  # time signature: estimation based on beats
  beats_per_bar = 4  # Default 4/4 
  if features.containsKey('rhythm.beats_count') and features['rhythm.beats_count'] > 16:
    beats_loudness = features['rhythm.beats_loudness'] if features.containsKey('rhythm.beats_loudness') else []
    if len(beats_loudness) >= 8: # accent beats
      autocorr = np.correlate(beats_loudness, beats_loudness, mode='full')
      middle = len(autocorr) // 2
      candidates = autocorr[middle+1:middle+5]
      beats_per_bar = np.argmax(candidates) + 2
  
  spotify_features["time_signature"] = beats_per_bar
  
  # valence [0, 1]
  if features.containsKey('highlevel.mood_happy') and 'happy' in features['highlevel.mood_happy']:
    spotify_features["valence"] = min(max(features['highlevel.mood_happy']['happy'], 0.0), 1.0)
  else: # fallback: HPCP entropy and mode
    hpcp_mean = features['tonal.hpcp.mean'] if features.containsKey('tonal.hpcp.mean') else [0.5] * 12
    hpcp_mean = np.mean(hpcp_mean)
    mode_factor = 0.6 if spotify_features["mode"] == 1 else 0.4  # naive assumption: songs in major scale sounds more happier
    spotify_features["valence"] = min(max(hpcp_mean * 10.0 * mode_factor, 0.0), 1.0)
  
  return spotify_features


def download_yt_audio(yt_id):
  yt_url = f"https://www.youtube.com/watch?v={yt_id}"

  ydl_opts = {
    'format': 'bestaudio/best', 
    'outtmpl': str(TMP / '%(id)s.%(ext)s'),
    'postprocessors': [{
      'key': 'FFmpegExtractAudio',
      'preferredcodec': 'mp3',
      'preferredquality': '192',
    }],
    'quiet': True,
    'no_warnings': True
  }
    
  with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(yt_url, download=False)
    if info.get('is_live ', False) or info.get('was_live', False):
      return None
    _ = ydl.download([yt_url])
  
  downloaded_file = TMP / f"{yt_id}.mp3"
  
  return downloaded_file



def main(csv_file, start_idx):
  ts = get_ts()

  wandb_run = wandb.init(
    entity='halsoo',
    project='rrs-crawl', 
    name=f'{ts}-audio_features',
  )

  # load city list for nordvpn
  with open(CWD / 'city_list_fast.txt', 'r') as f:
    city_list = f.readlines()
  city_list = [ c.rstrip() for c in city_list ]
  city_list = random.sample(city_list, len(city_list))
  city_iter = iter(city_list)

  df = pd.read_csv( csv_file )
  df = df.iloc[start_idx:]
  print("# of rows:", len(df))

  # create csv file for audio features
  output_path = f'track_yt_audio_features_{ts}.csv'
  output_path = CWD / 'data' / output_path
  with open(output_path, 'a') as f: 
    output_writer = csv.writer(f)
    output_writer.writerow(['artist', 'album', 'track', 'yt_id', 'yt_title', 'duration_seconds', 'score', 'acousticness', 'danceability', 'energy', 'speechiness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'tempo', 'time_signature', 'valence'])

  error_path = output_path.with_suffix('.log')
  error_log = open(error_path, 'a')

  total_saved = 0
  cur_city = ''

  total_results = []

  pbar = tqdm(
    df.iterrows(), 
    total=len(df), 
    dynamic_ncols=True, 
    postfix={'city': cur_city, 'results': total_saved }
  )

  for i, (artist, album, track, yt_id, yt_title, _, duration_seconds, score) in pbar:
    # change city every 100 downloads
    if i % VPN_CHANGE_INTERVAL == 0 or i == START_IDX:
      cur_city, city_iter = change_location(city_list, city_iter)
      pbar.set_postfix({'city': cur_city, 'results': total_saved })
    
    retry_cnt = 0
    
    while retry_cnt < 3:
      try:
        downloaded_file = download_yt_audio(yt_id)
        if downloaded_file is None:
          break
        
        feature_dict = extract_audio_features(str(downloaded_file))
        
        downloaded_file.unlink(missing_ok=True) # remove temp file
  
        feature_list = [
          feature_dict[k] 
          for k in ['acousticness', 'danceability', 'energy', 'speechiness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'tempo', 'time_signature', 'valence'] 
        ]
        
        row = [artist, album, track, yt_id, yt_title, duration_seconds, score] + feature_list
        total_results.append(row)
        break
      
      except Exception as e:
        if "Sign in to confirm you’re not a bot." in str(e):
          cur_city, city_iter = change_location(city_list, city_iter)
          pbar.set_postfix({'city': cur_city, 'results': total_saved })
          retry_cnt += 1
          continue
        
        else:
          print(f"{yt_id}: {yt_title}: {e}", file=error_log)  
          break
    
    # write result to file every 20 tracks
    if len(total_results) > VPN_CHANGE_INTERVAL - 1:
      total_saved += len(total_results)
      pbar.set_postfix({'city': cur_city, 'results': total_saved })
      wandb_run.log({'num_saved': total_saved})
      
      with open(output_path, 'a') as f: 
        output_writer = csv.writer(f)
        output_writer.writerows(total_results)

      total_results = []

    sleep(0.1)
  
  error_log.close()
  print("Done!")


if __name__ == '__main__':
  main(CSV_FILE, START_IDX)