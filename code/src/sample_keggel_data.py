'''
script selects a random sample of 1000 tracks from the keggel spotify data and creats a data set of them
'''
import random
import pandas as pd
import numpy as np
import spotipy
from tqdm import tqdm


def creats_keggel_random_sample(keggel_tracks):
    """
    generates a random sample of size 1000 from all songs in the keggel data set
    :param keggel_tracks:
    """
    random.seed(69)
    idx = np.arange(0, len(keggel_tracks))
    sample_idx = random.sample(list(idx), 1000)
    sample_keggel = keggel_tracks.iloc[sample_idx]
    return sample_keggel

def get_popularity(keggel_spotify):
    """
    loads popularity measure for given song ids and append them to the track data
    :param keggel_spotify: DataFrame of keggel spotify tracks
    :return: DataFrame of keggel spotify tracks with popularity column
    """
    # connect to spotify api
    cid = '020f2ed496864af6ab98a08846eac5da'
    secret = 'abd17ef2d04d4d3abd706ec03db71646'
    client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    popularity = []
    # get popularity of each track
    for idx in tqdm(keggel_spotify.id):
        popularity.append((sp.track(idx)['popularity']))
    keggel_spotify['popularity'] = popularity
    return keggel_spotify.reset_index(drop=True)

def main():
    keggel_tracks = pd.read_csv('../dat/tracks_features.csv', header=0)
    sample_keggel = creats_keggel_random_sample(keggel_tracks)
    sample_keggel = get_popularity(sample_keggel)
    print(sample_keggel)
    sample_keggel.to_csv('../dat/sample_keggel_tracks.csv',index=False)