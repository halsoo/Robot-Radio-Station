import os
from collections import defaultdict, Counter

from tqdm.auto import tqdm

import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix


def get_average_embedding(text, word_vectors):
    tokens = text.lower().split()
    vectors = []
    
    for token in tokens:
      if token in word_vectors:
        vectors.append(word_vectors[token])
    
    if vectors:
      return np.mean(vectors, axis=0)
    
    else: # fallback
      return np.zeros(word_vectors.vector_size)



def extract_feature_views(playlists:list, uniq_tracks:list, word_vectors):
  """
  - audio: audio features from Essentia
  - metadata: artist, album, track, duration, popularities
  - playlist: co-occurrence patterns
  """
  audio_features = {}
  metadata_features = {}
  playlist_features = {}
  
  for d in tqdm(uniq_tracks):
    track_uri, artist, album, track, yt_id, yt_title, duration_seconds, score, acousticness, danceability, speechiness, instrumentalness, key, liveness, loudness, mode, tempo, valence, artist_popularity, album_popularity, track_popularity = d
    
    audio_features[track_uri] = {
      'acousticness': float(acousticness) if float(acousticness) >= 0 else np.nan,
      'danceability': float(danceability) if float(danceability) >= 0 else np.nan,
      'speechiness': float(speechiness) if float(speechiness) >= 0 else np.nan,
      'instrumentalness': float(instrumentalness) if float(instrumentalness) >= 0 else np.nan,
      'key': float(key),
      'liveness': float(liveness) if float(liveness) >= 0 else np.nan,
      'loudness': float(loudness) if loudness != '' else np.nan,
      'mode': float(mode) if float(mode) >= 0 else np.nan,
      'tempo': float(tempo) if float(tempo) >= 0 else np.nan,
      'valence': float(valence) if float(valence) >= 0 else np.nan,
    }
    
    metadata_features[track_uri] = {
      'artist': get_average_embedding(artist, word_vectors),
      'album': get_average_embedding(album, word_vectors),
      'track': get_average_embedding(track, word_vectors),
      'duration': float(duration_seconds),
      'artist_popularity': float(artist_popularity), 
      'album_popularity': float(album_popularity), 
      'track_popularity': float(track_popularity),
    }
  
  song_to_playlists = defaultdict(set)
  
  for pl_i, pl in enumerate(tqdm(playlists)):
    for t in pl['tracks']:
      track_uri = t['track_uri']
      song_to_playlists[track_uri].add(pl_i)
  
  # co-occurrence matrix
  uniq_track_uri_list = [ track_uri for track_uri, *_ in uniq_tracks ]
  track_uri_to_idx = { track_uri: idx for idx, track_uri in enumerate(uniq_track_uri_list) }
  
  # Use SVD to reduce dimensionality of co-occurrence patterns
  if len(uniq_track_uri_list) > 0:
    # sparse co-occurrence matrix
    rows, cols, data = [], [], []
    for track_uri, playlists in song_to_playlists.items():
      if track_uri in track_uri_to_idx:
        for pl_i in playlists:
          rows.append(track_uri_to_idx[track_uri])
          cols.append(pl_i)
          data.append(1.0)
    
    # Create sparse matrix
    if len(rows) > 0:
      track_playlist_matrix = csr_matrix(
        (data, (rows, cols)), 
        shape=(len(uniq_track_uri_list), max(cols) + 1 if cols else 0)
      )
      
      # SVD
      n_components = min(100, track_playlist_matrix.shape[1] - 1)
      if n_components > 0:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        playlist_vectors = svd.fit_transform(track_playlist_matrix)
        
        # Convert to features
        for i, track_uri in enumerate(uniq_track_uri_list):
          features = {
            f"playlist_{j}": float(playlist_vectors[i, j]) 
            for j in range(playlist_vectors.shape[1])
          }
          playlist_features[track_uri] = features
  
  return audio_features, metadata_features, playlist_features



def prepare_feature_matrix(feature_dict, imputer='simple', is_metadata=False):
  # feature_dict => matrix
  # all feature vectors have must have the same dimensions
  
  if not feature_dict:
    return np.array([])
  
  # all unique keys
  all_keys = set()
  for features in tqdm(feature_dict.values()):
    all_keys.update(features.keys())
  
  all_keys = sorted(list(all_keys))
  
  # create feature matrix
  feature_matrix = []
  track_uri_list = []
  
  for track_uri, features in tqdm(feature_dict.items()):
    if is_metadata:
      vector = []
      vector += features['artist'].tolist() # word vector
      vector += features['album'].tolist() # word vector
      vector += features['track'].tolist() # word vector
      vector += [
        features['duration'], # scalar
        features['artist_popularity'], # scalar
        features['album_popularity'], # scalar
        features['track_popularity'] # scalar
      ]
    
    else:
      vector = [ features.get(key, np.nan) for key in all_keys ]
    
    feature_matrix.append(vector)
    track_uri_list.append(track_uri)
  
  feature_matrix = np.array(feature_matrix)
  
  # apply imputation for np.nan
  if np.isnan(feature_matrix).any():
    if imputer == 'simple':
      imputer = SimpleImputer(strategy='mean')  # or 'median', 'most_frequent', 'constant'
      feature_matrix = imputer.fit_transform(feature_matrix)
    
    elif imputer == 'knn':
      imputer = KNNImputer(n_neighbors=5)
      feature_matrix = imputer.fit_transform(feature_matrix)
  
  return feature_matrix, all_keys



def fit_clustering_model(track_ids, feature_matrix, feature_keys, num_clusters, use_minibatch=True, is_metadata=False):
  if is_metadata:
    embeddings = feature_matrix[:, :192]  # 3 embeddings of 64 dims each
    scalars = feature_matrix[:, 192:]     # 4 scalar features
  
    scaler = StandardScaler()
    scaled_scalars = scaler.fit_transform(scalars)
    
    scaled_features = np.hstack((embeddings, scaled_scalars))
    
  else:
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)
  
  if use_minibatch:
    n_clusters = min(num_clusters, len(track_ids))
      
    mini_kmeans = MiniBatchKMeans(
      n_clusters=n_clusters, 
      batch_size=25000, 
      max_iter=100,
      init='k-means++', 
      n_init=3,  
      init_size=2000*3,
      random_state=42, 
    )
    mini_kmeans.fit(scaled_features)

    clustering = KMeans(
      n_clusters=n_clusters, 
      init=mini_kmeans.cluster_centers_, 
      n_init=1, 
      max_iter=30, 
      random_state=42,
    )
    labels = clustering.fit_predict(scaled_features)
  
  else:
    n_clusters = min(num_clusters, 1000)
    
    clustering = KMeans(
      n_clusters=n_clusters, 
      init='k-means++', 
      n_init=10, 
      random_state=42,
    )
    labels = clustering.fit_predict(scaled_features)
  
  clusters = {
    t_id: int(label) 
    for t_id, label in zip(track_ids, labels)
  }
  
  return (
    clusters, 
    {
      'model': clustering,
      'scaler': scaler,
      'feature_keys': feature_keys
    }
  )



def multi_view_clustering(
  audio_features, 
  metadata_features, 
  playlist_features, 
  num_clusters=2000
):
  view_clusters = {}
  view_models = {}
  
  # audio view
  if audio_features:
    audio_ids = list(audio_features.keys())
    feature_matrix, feature_keys = prepare_feature_matrix(audio_features)
    
    if len(feature_matrix) > 0:
      clusters, model = fit_clustering_model(
        audio_ids, feature_matrix, feature_keys, num_clusters
      )
      view_clusters['audio'] = clusters
      view_models['audio'] = model
  
  # metadata view
  if metadata_features:
    metadata_ids = list(metadata_features.keys())
    feature_matrix, feature_keys = prepare_feature_matrix(metadata_features)
    
    if len(feature_matrix) > 0:
      clusters, model = fit_clustering_model(
        metadata_ids, feature_matrix, feature_keys, num_clusters
      )
      view_clusters['metadata'] = clusters
      view_models['metadata'] = model
  
  # playlist view
  if playlist_features:
    playlist_ids = list(playlist_features.keys())
    feature_matrix = prepare_feature_matrix(playlist_features)
    
    if len(feature_matrix) > 0:
      clusters, model = fit_clustering_model(
        playlist_ids, feature_matrix, feature_keys, num_clusters
      )
      view_clusters['playlist'] = clusters
      view_models['playlist'] = model
  
  return view_clusters, view_models



def integrate_view_clusters(uniq_track_uri_list, view_clusters):
  # hierarchical order (audio > metadata > playlist)
  # content-based approach
  final_clusters = {}
  
  for track_uri in uniq_track_uri_list:
    views = []
    if 'audio' in view_clusters and track_uri in view_clusters['audio']:
      views.append('audio')
    if 'metadata' in view_clusters and track_uri in view_clusters['metadata']:
      views.append('metadata')
    if 'playlist' in view_clusters and track_uri in view_clusters['playlist']:
      views.append('playlist')
    
    
    if 'audio' in views and 'metadata' in views:
      audio_cluster = view_clusters['audio'][track_uri]
      metadata_cluster = view_clusters['metadata'][track_uri]
      
      # audio in high bits, metadata in middle bits
      combined_cluster = (
        (audio_cluster & 0x3FF) << 20) | ((metadata_cluster & 0x3FF) << 10
      )
      
      # playlist in low bits
      if 'playlist' in views:
        playlist_cluster = view_clusters['playlist'][track_uri]
        combined_cluster |= (playlist_cluster & 0x3FF)
      
      final_clusters[track_uri] = combined_cluster
    
    # only audio available
    elif 'audio' in views:
      audio_cluster = view_clusters['audio'][track_uri]
      if 'playlist' in views:
        playlist_cluster = view_clusters['playlist'][track_uri]
        final_clusters[track_uri] = (
          ((audio_cluster & 0xFFFF) << 16) | (playlist_cluster & 0xFFFF)
        )
      else:
        final_clusters[track_uri] = audio_cluster
    
    # only metadata available
    elif 'metadata' in views:
      metadata_cluster = view_clusters['metadata'][track_uri]
      if 'playlist' in views:
        playlist_cluster = view_clusters['playlist'][track_uri]
        final_clusters[track_uri] = (
          ((metadata_cluster & 0xFFFF) << 16) | (playlist_cluster & 0xFFFF)
        )
      else:
        final_clusters[track_uri] = metadata_cluster
    
    # only playlist available
    elif 'playlist' in views:
      final_clusters[track_uri] = view_clusters['playlist'][track_uri]
    
    # no features available
    else:
      final_clusters[track_uri] = 0
  
  
  cluster_to_tracks = defaultdict(list)
  for track_uri, cluster in final_clusters.items():
    cluster_to_tracks[cluster].append(track_uri)
  
  track_to_cluster = final_clusters
  cluster_to_tracks = dict(cluster_to_tracks)
  
  return track_to_cluster, dict(cluster_to_tracks)



def estimate_new_song_popularity(
  new_song_metadata, 
  track_to_artist,
  existing_popularity_features,
):
  # default values
  estimated_features = {
    'song_popularity': 0.01, 
    'artist_popularity': 0.0,
    'album_popularity': 0.0,
    'combined_popularity': 0.01
  }
  
  
  if 'artist' in new_song_metadata:
    artist = new_song_metadata['artist']
    
    artist_tracks = []
    for track_uri, features in existing_popularity_features.items():
      track_artist = track_to_artist.get(track_uri)
      if track_artist == artist:
        artist_tracks.append(track_uri)
    
    if artist_tracks:
      avg_artist_pop = np.mean([
        existing_popularity_features[t]['artist_popularity'] 
        for t in artist_tracks
      ])  
      estimated_features['artist_popularity'] = avg_artist_pop
      
      # Update combined popularity
      estimated_features['combined_popularity'] = (
        estimated_features['song_popularity'] * 0.5 +
        avg_artist_pop * 0.3 + 
        estimated_features['album_popularity'] * 0.2
      )
  
  return estimated_features



def assign_new_song_to_cluster(
  new_track_uri,
  new_audio_feature, 
  new_metadata_feature,
  audio_features,
  metadata_features, 
  view_models, 
  existing_view_clusters,
  vocab,
):
  view_assignments = {}
  
  if new_audio_feature and 'audio' in view_models:
    audio_model = view_models['audio']['model']
    audio_scaler = view_models['audio']['scaler']
    feature_keys = view_models['audio']['feature_keys']
    
    vector = [
      new_audio_feature.get(key, 0.0) 
      for key in feature_keys
    ]
    scaled_vector = audio_scaler.transform([vector])
    
    audio_cluster = int(audio_model.predict(scaled_vector)[0])
    view_assignments['audio'] = audio_cluster
  
  
  if new_metadata_feature and 'metadata' in view_models:
    metadata_model = view_models['metadata']['model']
    metadata_scaler = view_models['metadata']['scaler']
    feature_keys = view_models['metadata']['feature_keys']
    
    vector = [
      new_metadata_feature.get(key, 0.0)
      for key in feature_keys
    ]
    scaled_vector = metadata_scaler.transform([vector])
    
    metadata_cluster = int(metadata_model.predict(scaled_vector)[0])
    view_assignments['metadata'] = metadata_cluster
  
  
  # approximate playlist view
  if (
    'playlist' in view_models and
    (
      'audio' in view_assignments or
      'metadata' in view_assignments
    )
  ):
    if 'audio' in view_assignments:
      primary_view = 'audio'
    else:
      primary_view = 'metadata'
    
    primary_cluster = view_assignments[primary_view]
    
    similar_tracks = [
      track_uri 
      for track_uri, cluster in existing_view_clusters[primary_view].items() 
      if cluster == primary_cluster
    ]
    
    if (
      similar_tracks and 
      'playlist' in existing_view_clusters
    ):
      playlist_clusters = [
        existing_view_clusters['playlist'].get(track_uri) 
        for track_uri in similar_tracks 
        if track_uri in existing_view_clusters['playlist']
      ]
      
      if playlist_clusters:
        most_common = Counter(playlist_clusters).most_common(1)[0][0]
        view_assignments['playlist'] = most_common
  
  
  if 'audio' in view_assignments and 'metadata' in view_assignments:
    audio_cluster = view_assignments['audio']
    metadata_cluster = view_assignments['metadata']
    
    combined_cluster = (
      ((audio_cluster & 0x3FF) << 20) | ((metadata_cluster & 0x3FF) << 10)
    )

    if 'playlist' in view_assignments:
      playlist_cluster = view_assignments['playlist']
      combined_cluster |= (playlist_cluster & 0x3FF)
    
    final_cluster = combined_cluster
  
    
  # audio only
  elif 'audio' in view_assignments:
    audio_cluster = view_assignments['audio']
    
    if 'playlist' in view_assignments:
      playlist_cluster = view_assignments['playlist']
      final_cluster = (
        ((audio_cluster & 0xFFFF) << 16) | (playlist_cluster & 0xFFFF)
      )
    else:
      final_cluster = audio_cluster
  
  
  # metadata only
  elif 'metadata' in view_assignments:
    metadata_cluster = view_assignments['metadata']
    if 'playlist' in view_assignments:
      playlist_cluster = view_assignments['playlist']
      final_cluster = (
        ((metadata_cluster & 0xFFFF) << 16) | (playlist_cluster & 0xFFFF)
      )
    else:
      final_cluster = metadata_cluster
  
  
  # playlist only (???)
  elif 'playlist' in view_assignments:
    final_cluster = view_assignments['playlist']
  
  
  # fallback
  else:
    final_cluster = 0
  
  
  # get position in cluster for the new song
  combined_feature_dict = {}
  for existing_track_uri in vocab.track_to_cluster:
    existing_combined = {}
    if existing_track_uri in audio_features:
      existing_combined.update(audio_features[existing_track_uri])
    if existing_track_uri in metadata_features:
      existing_combined.update(metadata_features[existing_track_uri])
    
    combined_feature_dict[existing_track_uri] = existing_combined
  
  # combined features for similarity calculation
  combined_features = {}
  if audio_features:
    combined_features.update(audio_features)
  if metadata_features:
    combined_features.update(metadata_features)
  
  # Add to vocabulary using similarity-based position
  position = vocab.add_new_song_by_similarity(
    new_track_uri, final_cluster, combined_features, combined_feature_dict
  )
  
  return final_cluster, position