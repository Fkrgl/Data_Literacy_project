"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to download the top 50 chart playlists to a .csv-file.
"""
# --------------------------------------------------- PACKAGES ------------------------------------------------------- #
import spotipy
import pandas as pd

from tqdm import tqdm


# ---------------------------------------------- FUNCTIONS/UTILITY --------------------------------------------------- #
def call_playlists(playlist_links, playlist_countries, playlist_continents):
    """
    Iteratively calls data for a list of playlist-links and appends the data to a pd.DataFrame with additional
    columns for country & continent.

    This function was adapted for our data collecting purposes from:
    https://www.linkedin.com/pulse/extracting-your-fav-playlist-info-spotifys-api-samantha-jones

    Parameters
    ----------
    playlist_links : list(str)
        Contains final parts of the spotify playlist link, which is the playlist id.
        Example: 37i9dQZEVXbLn7RQmT5Xv2 corresponds to "Top 50 - Egypt"
    playlist_countries: list(str)
        IOC country codes of the respective playlist
        (https://de.wikipedia.org/wiki/Liste_der_IOC-L%C3%A4ndercodes)
    playlist_continents: list(str)
        Continent of the respective playlist

    Returns
    ----------
    pd.DataFrame containing the following information for all playlists:
    "artist","album","track_name", "track_id","danceability", "energy","key",
    "loudness","mode", "speechiness", "instrumentalness", "liveness","valence",
    "tempo", "duration_ms", "time_signature"
    """
    # Spotify API authentication
    cid = '020f2ed496864af6ab98a08846eac5da'
    secret = 'abd17ef2d04d4d3abd706ec03db71646'

    client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=cid, client_secret=secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # Track metadata and features that will be analyzed in this project
    track_data = ["artist", "album", "track_name", "track_id"]
    features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "instrumentalness",
                "liveness", "valence", "tempo", "duration_ms", "time_signature"]

    playlist_df = pd.DataFrame(columns=track_data + features + ["country", "continent"])
    for country_data in tqdm(zip(playlist_links, playlist_countries, playlist_continents), total=len(playlist_links)):
        chart_playlist = sp.user_playlist_tracks("spotify", country_data[0])["items"]

        # For each track, iteratively get track metadata, features and add to playlist_df in the end
        for track in chart_playlist:
            track_features = {}  # will represent a single line for the track in playlist_df
            # Metadata:
            track_features["artist"] = track["track"]["album"]["artists"][0]["name"]
            track_features["album"] = track["track"]["album"]["name"]
            track_features["track_name"] = track["track"]["name"]
            track_features["track_id"] = track["track"]["id"]
            track_features["popularity"] = track["track"]["popularity"]

            # Features:
            audio_features = sp.audio_features(track_features["track_id"])[0]
            if audio_features is None:
                # Apparently, some track do not have features available. These tracks will be skipped.
                print("NOTE: Features for song '{}' by '{}' are not available, hence the song will be skipped.".
                      format(track_features["track_name"], track_features["artist"]))
                continue

            for feat in features:
                track_features[feat] = audio_features[feat]

            # Geographical info:
            track_features["country"] = country_data[1]
            track_features["continent"] = country_data[2]

            # Concatenate line to playlist_df
            track_df = pd.DataFrame(track_features, index=[0])
            playlist_df = pd.concat([playlist_df, track_df], ignore_index=True)

    return playlist_df


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main(playlist_links, country_codes, continent_codes):
    """
    :param playlist_links: File path to spotify_links.txt
    :param country_codes: File path to spotify_countries.txt
    :param continent_codes: File path to spotify_continents.txt
    :return: pd.DataFrame containing features of the downloaded playlists
    """
    # Read playlist information:
    with open(playlist_links) as f:
        links = f.read().splitlines()
    with open(country_codes) as f:
        countries = f.read().splitlines()
    with open(continent_codes) as f:
        continents = f.read().splitlines()

    # Download playlist and save as .csv
    print("     ...downloading data of top 50 charts playlists for {} countries...".format(len(links)))
    data_output = '../dat/charts_data.csv'
    playlist_data = call_playlists(playlist_links=links, playlist_countries=countries, playlist_continents=continents)
    playlist_data.to_csv(data_output, index=None)
    print("     ...saved chart playlist data under '{}'...".format(data_output))

    # Additionally, remove all songs that occur in multiple playlists and also save to .csv
    data_output = '../dat/charts_data_no_duplicates.csv'
    playlist_data.drop_duplicates(subset=["track_id"], keep=False, inplace=True)
    playlist_data.to_csv(data_output, index=None)
    print("     ...saved chart playlist data for unique songs under '{}'...".format(data_output))


if __name__ == "__main__":
    main()
