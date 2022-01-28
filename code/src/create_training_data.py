import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import spotipy
import tqdm


def pre_process_data(data):
    """
    function drops duplicated rows, removes
    :param data:
    :return:
    """
    data.drop_duplicates(subset=['track_id'], keep=False, inplace=True)
    # features that are not required or should be excluded from standard scaling
    cat_features = ['artist', 'album', 'track_name', 'track_id', 'country', 'continent', 'popularity'
                ,'key', 'mode', 'time_signature']
    feature_data = np.asarray(data.drop(cat_features, axis = 1))
    column_names = feature_data.columns
    scaled_data = StandardScaler().fit_transform(feature_data)
    return scaled_data

def merge_keggel_and_chart_tracks(keggel_tracks, chart_tracks):
    # drop meta data and duplicates
    chart_tracks.drop_duplicates(subset=['track_id'], keep=False, inplace=True)
    metadata = ['artist', 'album', 'track_name', 'track_id', 'country',
                'continent']  # features that are actually not required
    chart_tracks = chart_tracks.drop(metadata, axis=1)
    # get shared features
    keggel_tracks_col = set(keggel_tracks.columns)
    chart_tracks_cols = set(chart_tracks.columns)
    intersection = list(chart_tracks_cols.intersection(keggel_tracks_col))
    # append datasets
    training_keggel = keggel_tracks[intersection]
    training_charts = chart_tracks[intersection]
    regression_data = training_charts.append(training_keggel)
    # separate features and labels
    y = regression_data['popularity'].reset_index(drop=True)
    regression_data = regression_data.drop('popularity', axis=1).reset_index(drop=True)
    return y, regression_data


def standardize_data(regression_data):
    """
    standardizes all numerical, not categorial features
    :param regression_data: merged data set
    :return: scaled merged data set
    """
    # exclude categotial varibles from scaling
    cat_features = ['key', 'mode', 'time_signature']
    categorial_columns = regression_data[cat_features].astype(str).reset_index(drop=True)
    regression_data = regression_data.drop(cat_features, axis=1)
    # scale data
    scaled_regression_data = StandardScaler().fit_transform(regression_data)
    scaled_regression_data = pd.DataFrame(scaled_regression_data)
    scaled_regression_data.columns = regression_data.columns
    # merge with categorial data
    scaled_regression_data[cat_features] = categorial_columns
    return scaled_regression_data

def main():
    # read in data
    chart_tracks = pd.read_csv('../dat/data_no_duplicates.csv')
    keggel_tracks = pd.read_csv('../dat/sample_keggel_tracks.csv')
    y, merged_data = merge_keggel_and_chart_tracks(keggel_tracks, chart_tracks)
    trainings_data = standardize_data(merged_data)
    print(trainings_data)
    y.to_csv('../dat/labels.csv', index=False)
    trainings_data.to_csv('../dat/trainings_data.csv', index=False)
main()

