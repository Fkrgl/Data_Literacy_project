"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to randomly sample 1000 tracks from the kaggle data set and to store it as .csv-file.
"""
# --------------------------------------------------- PACKAGES ------------------------------------------------------- #
import random
import pandas as pd
import numpy as np
import spotipy
from tqdm import tqdm


# ---------------------------------------------- FUNCTIONS/UTILITY --------------------------------------------------- #
def create_kaggle_random_sample(kaggle_tracks):
    """
    generates a random sample of size 1000 from all songs in the kaggle data set
    :param kaggle_tracks: Full data set downloaded from kaggle (1.2M+ songs)
    """
    random.seed(69)
    idx = np.arange(0, len(kaggle_tracks))
    sample_idx = random.sample(list(idx), 1000)
    sample_kaggle = kaggle_tracks.iloc[sample_idx]
    return sample_kaggle


def get_popularity(kaggle_spotify):
    """
    loads popularity measure for given song ids and append them to the track data
    :param kaggle_spotify: DataFrame of kaggle spotify tracks
    :return: DataFrame of kaggle spotify tracks with popularity column
    """
    # connect to spotify api
    cid = '020f2ed496864af6ab98a08846eac5da'
    secret = 'abd17ef2d04d4d3abd706ec03db71646'
    client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    popularity = []
    # get popularity of each track
    print("     ...extracting popularity for sampled tracks from spotify API...")
    for idx in tqdm(kaggle_spotify.id):
        popularity.append((sp.track(idx)['popularity']))
    kaggle_spotify['popularity'] = popularity
    return kaggle_spotify.reset_index(drop=True)


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main():
    pd.options.mode.chained_assignment = None
    print("Sampling Kaggle data set")
    kaggle_tracks = pd.read_csv('../dat/kaggle_data.csv', header=0)
    sample_kaggle = create_kaggle_random_sample(kaggle_tracks)
    sample_kaggle = get_popularity(sample_kaggle)
    sample_kaggle.to_csv('../dat/sample_kaggle_tracks.csv', index=False)
    print("     ...completed...")


if __name__ == "__main__":
    main()