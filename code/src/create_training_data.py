"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to preprocess the chart data set and to merge it with the random kaggle sample.
"""

# --------------------------------------------------- IMPORTS -------------------------------------------------------- #
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------- FUNCTIONS/UTILITY --------------------------------------------------- #
def pre_process_data(data):
    """
    function drops duplicated rows, removes
    :param data:
    :return:
    """
    data.drop_duplicates(subset=['track_id'], keep=False, inplace=True)
    # features that are not required or should be excluded from standard scaling
    cat_features = ['artist', 'album', 'track_name', 'track_id', 'country', 'continent', 'popularity',
                    'key', 'mode', 'time_signature']
    feature_data = np.asarray(data.drop(cat_features, axis=1))
    column_names = feature_data.columns
    scaled_data = StandardScaler().fit_transform(feature_data)
    return scaled_data


def merge_kaggle_and_chart_tracks(kaggle_tracks, chart_tracks):
    # drop meta data and duplicates
    print("     ...dropping duplicated tracks and metadata columns in charts data set...")
    chart_tracks.drop_duplicates(subset=['track_id'], keep=False, inplace=True)
    metadata = ['artist', 'album', 'track_name', 'track_id', 'country',
                'continent']  # features that are actually not required
    chart_tracks = chart_tracks.drop(metadata, axis=1)
    # get shared features
    kaggle_tracks_col = set(kaggle_tracks.columns)
    chart_tracks_cols = set(chart_tracks.columns)
    intersection = list(chart_tracks_cols.intersection(kaggle_tracks_col))
    # append datasets
    training_kaggle = kaggle_tracks[intersection]
    training_charts = chart_tracks[intersection]
    print("     ...merging charts and kaggle data sets...")
    regression_data = training_charts.append(training_kaggle)
    # separate features and labels
    y = regression_data['popularity'].reset_index(drop=True)
    regression_data = regression_data.drop('popularity', axis=1).reset_index(drop=True)
    return y, regression_data


def standardize_data(regression_data):
    """
    standardizes all numerical, not categorical features
    :param regression_data: merged data set
    :return: scaled merged data set
    """
    # exclude categorical variables from scaling
    cat_features = ['key', 'mode', 'time_signature']
    categorical_columns = regression_data[cat_features].astype(str).reset_index(drop=True)
    regression_data = regression_data.drop(cat_features, axis=1)
    # scale data
    scaled_regression_data = StandardScaler().fit_transform(regression_data)
    scaled_regression_data = pd.DataFrame(scaled_regression_data)
    scaled_regression_data.columns = regression_data.columns
    # merge with categorical data
    scaled_regression_data[cat_features] = categorical_columns
    return scaled_regression_data


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main():
    # read in data
    chart_tracks = pd.read_csv('../dat/charts_data_no_duplicates.csv')
    kaggle_tracks = pd.read_csv('../dat/sample_kaggle_tracks.csv')
    print("Transforming data")
    y, merged_data = merge_kaggle_and_chart_tracks(kaggle_tracks, chart_tracks)
    training_data = standardize_data(merged_data)
    y.to_csv('../dat/response.csv', index=False)
    print("     ...Saved response under '../dat/response.csv'...")
    training_data.to_csv('../dat/training_data.csv', index=False)
    print("     ...Saved training data under '../dat/training_data.csv'...")


if __name__ == "__main__":
    main()
