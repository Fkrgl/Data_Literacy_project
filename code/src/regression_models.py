'''
this script performs linear and logistic regression on pre-selected song features to predict the popularity of a song
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression

def get_root_mean_squared_error(predictions, labels):
    rsme = 0
    for pred, label in zip(predictions, labels):
        rsme += (label - pred)**2
    return np.sqrt(1/len(predictions) * rsme)

def print_linear_regression_performance(reg, X_train, X_test, y_train, y_test):
    '''
    prints the trainings, CV and test RSME for a trained linear regression model
    '''
    predictions = reg.predict(X_train)
    rsme_normal = np.sqrt(mean_squared_error(y_train, predictions))
    print(f'train error: {rsme_normal}')

    # cross validation
    model = LinearRegression()
    scorer = make_scorer(get_root_mean_squared_error)
    cv = KFold(n_splits=5, shuffle=True)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scorer)
    print(f'mean cross validation error: {scores.mean()}')

    # get test error
    predictions_training = reg.predict(X_test)
    print(f'test error: {np.sqrt(mean_squared_error(y_test, predictions_training))}')

def print_log_regression_performance(clf, X_train, X_test, y_train, y_test):
    print(f'trainings score: {clf.score(X_train, y_train)}')
    print(f'test score: {clf.score(X_test, y_test)}')

def train_linear_regression_model(X_train, y_train):
    # perform linear regerssion and evaluate
    reg = LinearRegression().fit(X_train, y_train)
    return reg

def train_logistic_regression_model(X_train, y_train):
    clf = LogisticRegression().fit(X_train, y_train)
    return clf

def get_logistic_regression_data_sets(y, trainings_data):
    """
    binaryizes popularity by setting all values above 50 to one, all other to zero. The a split of the data in test and
    trainings set is perfomred
    :param y: labels
    :param trainings_data: data
    :return: trainings and test data sets
    """
    y_logistic = np.zeros(len(y))
    y_logistic[y > 50] = 1
    # split data into test and trainings set
    X_train, X_test, y_train, y_test = train_test_split(trainings_data, y_logistic, test_size=0.33,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def create_four_fearure_data_set(training_data):
    """
    extracts the four most informative features from data and creats new training set with it
    """

    training_data_four_params = training_data[['danceability', 'energy', 'valence', 'loudness']]
    return training_data_four_params

def main():
    # read in data
    trainings_data = pd.read_csv('../dat/trainings_data.csv')
    y = pd.read_csv('../dat/labels.csv').values[:,0]

    # linear regression
    print('\nlinear regression:')
    X_train, X_test, y_train, y_test = train_test_split(trainings_data, y, test_size=0.33, random_state=42)
    reg = train_linear_regression_model(X_train, y_train)
    print_linear_regression_performance(reg, X_train, X_test, y_train, y_test)

    # logistic regression
    print('\nlogistic regression:')
    X_train, X_test, y_train, y_test = get_logistic_regression_data_sets(y, trainings_data)
    log_reg = train_logistic_regression_model(X_train, y_train)
    print_log_regression_performance(log_reg, X_train, X_test, y_train, y_test)

    # logistic regression on four feature
    print('\nlogistic regression on four most informative features:')
    training_data_four_params = create_four_fearure_data_set(trainings_data)
    X_train, X_test, y_train, y_test = get_logistic_regression_data_sets(y, training_data_four_params)
    log_reg = train_logistic_regression_model(X_train, y_train)
    print_log_regression_performance(log_reg, X_train, X_test, y_train, y_test)


main()