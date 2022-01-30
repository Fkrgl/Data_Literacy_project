"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to apply dimension reduction using tSNE and to create plots with different color codings
(by data source and also feature values)
"""
# --------------------------------------------------- IMPORTS -------------------------------------------------------- #
import pandas as pd
import warnings
from plotnine import *

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main():
    print("Processing data for tSNE plots")
    # ======== Load and process data ======== #
    print("     ...loading data...")
    # load data
    playlist_data = pd.read_csv("../dat/histogram_data.csv")
    print("     ...selecting features...")
    playlist_data = playlist_data[['danceability', 'energy', 'valence', 'loudness', 'dataset', 'popularity']]
    # scale data
    print("     ...scaling data...")
    scaled_data = StandardScaler().fit_transform(playlist_data.drop(['dataset', 'popularity'], axis=1))

    # ======== tSNE ======== #
    print("Applying tSNE")
    print("     ...applying dimension reduction...")
    tSNE_embedding = pd.DataFrame(TSNE(n_components=2, learning_rate='auto',
                                       init='random', random_state=420).fit_transform(scaled_data))
    tSNE_embedding[playlist_data.columns] = playlist_data[playlist_data.columns]

    print("     ...preparing data for plot...")
    tSNE_embedding['popularity*10e-2'] = tSNE_embedding['popularity'] / 100
    tSNE_embedding.rename(columns={0: 'tSNE 1', 1: 'tSNE 2'}, inplace=True)
    features_of_interest = ['tSNE 1', 'tSNE 2', 'popularity*10e-2', 'danceability', 'valence', 'energy']
    plot_data = pd.melt(tSNE_embedding.loc[:, tSNE_embedding.columns.isin(features_of_interest)],
                        id_vars=['tSNE 1', 'tSNE 2'])

    warnings.filterwarnings("ignore", category=UserWarning)
    # ======== Plots ======== #
    print("Creating plots")
    # colored by data set
    p1 = (ggplot(data=tSNE_embedding) + aes(x='tSNE 1', y='tSNE 2', color='dataset') + geom_point() +
          theme(legend_position=(0.775, 0.8), legend_title=element_blank(),
                legend_background=element_rect(fill='white', color='white', alpha=0.6)))
    p1.save("../fig/tSNE_datasets.pdf", width=6, height=5)
    print("     ...saved tSNE plot colored by data source under ../fig/tSNE_datasets.pdf...")

    # colored by feature value
    p2 = (ggplot(data=plot_data) + aes(x='tSNE 1', y='tSNE 2', color='value') + geom_point(alpha=0.3) +
          facet_wrap('~variable'))
    p2.save("../fig/tSNE_features.pdf", width=6, height=6)
    print("     ...saved tSNE plot(s) colored by feature value under ../fig/tSNE_datasets.pdf...")


if __name__ == "__main__":
    main()
