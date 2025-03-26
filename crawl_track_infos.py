import sys
import subprocess
from pathlib import Path
from time import sleep
from datetime import datetime

import math
import random

import csv

import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import wandb

from yt_dlp import YoutubeDL

START_IDX = 36508
VPN_CHANGE_INTERVAL = 100

get_ts = lambda: datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


HOME = Path.home()
CWD = Path.cwd()


def search_youtube(
  query:str, 
  max_results:int=5, 
  download:bool=False, 
  cookie_path:str=None
) -> list:
  
  ydl_opts = {
    'format': 'bestaudio/best',
    'quiet': True,
    'no_warnings': True,
    'ignoreerrors': True,
    'extract_flat': 'in_playlist',
    'default_search': f'ytsearch{max_results}',
    'cookies': cookie_path,
    'skip_download': not download,
  }
  
  # Create YoutubeDL object
  with YoutubeDL(ydl_opts) as ydl:
    results = ydl.extract_info(query, download=download)
    if results is None:
      return []
    
    # Process results
    if 'entries' in results:
      processed_results = []
      
      for entry in results['entries']:
        if entry is None:
          continue
        
        try:
          # get more detailed infos
          video_info = ydl.extract_info(entry['url'], download=False)
          
          # extract relevant infos
          duration_seconds = video_info.get('duration', 0)
          channel_name = video_info.get('uploader', '')
          channel_id = video_info.get('channel_id', '')
          
          result = {
            'id': entry.get('id', ''),
            'title': entry.get('title', ''),
            'url': entry.get('url', ''),
            'channel_name': channel_name,
            'channel_id': channel_id,
            'duration_seconds': duration_seconds,
          }
          
          processed_results.append(result)
        
        except Exception as e:
          print(
            f"Error processing entry {entry.get('id', 'unknown')}: {e}", 
            file=sys.stderr
          )
      
      return processed_results
  
  return []


def filter_search_results(result:list, track_info:list, dur_tol=2) -> list:
  artist_i, _, track_i, dur_mean, dur_std = track_info
  dur_mean, dur_std = round(dur_mean / 1000), round(dur_std / 1000)
  
  filtered_results = []
  for r in result:
    artist_i, track_i = artist_i.lower(), track_i.lower()
    
    title_r, uploader_r, duration_r = r.get('title', ''), r.get('channel_name', ''), r.get('duration_seconds', None)
    title_r = title_r.lower() if title_r else ''
    uploader_r = uploader_r.lower() if uploader_r else ''
    
    if 'live' in title_r or 'cover' in title_r or 'remix' in title_r or 'instrumental' in title_r or 'karaoke' in title_r:
      continue
    
    is_right_dur = (
      dur_mean-dur_std <= duration_r <= dur_mean+dur_std
      if dur_std > 0 
      else dur_mean-dur_tol <= duration_r <= dur_mean+dur_tol
    )
    
    score = 0
    if is_right_dur:
      score += 1
    if artist_i in title_r:
      score += 1
    if track_i in title_r:
      score += 1
    if artist_i in uploader_r:
      score += 1
      
    if score > 0:
      r['score'] = score
      filtered_results.append(r)
  
  return filtered_results


if __name__ == '__main__':
  ts = get_ts()

  wandb_run = wandb.init(
    entity='halsoo',
    project='rrs-crawl', 
    name=f'{ts}',
  )

  with open(CWD / 'city_list_fast.txt', 'r') as f:
    city_list = f.readlines()
  city_list = [ c.rstrip() for c in city_list ]
  city_list = random.sample(city_list, len(city_list))
  city_iter = iter(city_list)

  df = pd.read_csv( CWD / 'data' / 'uniq_tracks.csv')
  df = df.iloc[START_IDX:]
  print("# of rows:", len(df))

  output_path = f'track_yt_infos_{ts}.csv'
  output_path = CWD / 'data' / output_path
  output_file = open(output_path, 'a')
  output_writer = csv.writer(output_file)
  output_writer.writerow(['artist', 'album', 'track', 'yt_id', 'yt_title', 'yt_channel_name', 'duration_seconds', 'score'])

  total_saved = 0
  cur_city = ''

  total_results = []

  pbar = tqdm(df.iterrows(), total=len(df), dynamic_ncols=True, postfix={'city': cur_city, 'results': total_saved }, initial=START_IDX)

  for i, (artist, album, track, duration_mean, duration_std) in pbar:
    # change city every 10_000 queries
    if i % VPN_CHANGE_INTERVAL == 0 or i == START_IDX:
      while True:
        try: 
          city = next(city_iter)
        except StopIteration:
          city_list = random.sample(city_list, len(city_list))
          city_iter = iter(city_list)
          city = next(city_iter)

        cmd = ['nordvpn', 'connect', city]
        try:
          subprocess.run(cmd, check=True)
          cur_city = city
          pbar.set_postfix({'city': cur_city, 'results': total_saved })
          break
        except:
          continue
    
    query = f'{artist.lower()} {track.lower()} official lyrics'
    query = query.replace(':', '\\:')
    result = search_youtube(
      query, 
      max_results=2, 
      download=False
    )
    
    if len(result) > 0:
      pbar.set_description(f"{len(result)} for {artist} - {track}")
      result = filter_search_results(result, (artist, album, track, duration_mean, duration_std))
    if len(result) > 0:
      result = max(result, key=lambda x: x['score'])
      total_results.append((artist, album, track, result['id'], result['title'], result['channel_name'], result['duration_seconds'], result['score']))
    
    # write result to file every 100 tracks
    if len(total_results) > VPN_CHANGE_INTERVAL - 1:
      total_saved += len(total_results)
      pbar.set_postfix({'city': cur_city, 'results': total_saved })
      wandb_run.log({'num_saved': total_saved})
      
      output_writer.writerows(total_results)

      total_results = []

    sleep(0.1)
  
  output_file.close()
  print("Done!")