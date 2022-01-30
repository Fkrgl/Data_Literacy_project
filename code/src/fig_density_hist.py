"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to create a density histogram comparing the distribution of the spotify charts playlists and
the random sample from Kaggle
"""
# --------------------------------------------------- IMPORTS -------------------------------------------------------- #
import pandas as pd
import warnings
from plotnine import *


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main():
    print("Processing data for density histograms")

    # ======== Load and process data ======== #
    # load data
    chart_data = pd.read_csv("../dat/charts_data_no_duplicates.csv")
    kaggle_data = pd.read_csv("../dat/sample_kaggle_tracks.csv")

    # drop categorical features and metadata:
    print("     ...dropping categorical features...")
    chart_metadata = ['artist', 'album', 'track_name', 'track_id', 'country', 'continent']
    kaggle_metadata = ['id', 'name', 'album', 'album_id', 'artists', 'acousticness', 'artist_ids', 'track_number',
                       'disc_number', 'explicit', 'year', 'release_date']
    cat_features = ['key', 'mode', 'time_signature']
    chart_data.drop(chart_metadata + cat_features, axis=1, inplace=True)
    kaggle_data.drop(kaggle_metadata + cat_features, axis=1, inplace=True)

    # add info about data set source
    print("     ...appending data source information...")
    chart_data['dataset'] = "Top 50 Charts"
    kaggle_data['dataset'] = "Kaggle Samples"
    training_data = chart_data.append(kaggle_data)

    # duration in ms --> duration in s
    training_data['duration (s)'] = training_data['duration_ms'] / 1000
    training_data.drop('duration_ms', axis=1, inplace=True)
    training_data.to_csv("../dat/histogram_data.csv")

    # melt in order to use ggplot2
    plot_data = pd.melt(training_data, id_vars='dataset')

    # ======== Create plot ======== #
    print("Creating plot")
    density = (ggplot(data=plot_data) +
               aes(x='value', fill='dataset', color='dataset') +
               geom_density(alpha=0.6) +
               facet_wrap('~variable', scales='free') +
               theme(subplots_adjust={'hspace': 0.4, 'wspace': 0.32}) +
               ggtitle("Distribution of chart data versus random sample"))

    warnings.filterwarnings("ignore", category=UserWarning)
    density.save("../fig/Charts_vs_random_samples.pdf", width=10, height=6)
    print("     ...saved density histograms under '../fig/Charts_vs_random_samples.pdf'...")


if __name__ == "__main__":
    main()
