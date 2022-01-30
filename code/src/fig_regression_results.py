"""
Data Literacy - WiSe 21
Project: Assessment of Spotify Chart Preferences across Different Countries
Authors: Florian Kriegel (5746680), Jonathan Waehrer (5776535)

Purpose of this script is to produce plots the summarize the results of the regression models. Following plots will be
created by this script:
- residual plot of linear regression model. Colored by 'danceability'
- Histogram of popularity. Colored by predcited vs. true levels
- Confusion matrix of prediction vs true class resulting from logistic regression model
"""
# --------------------------------------------------- IMPORTS -------------------------------------------------------- #
import pandas as pd
import warnings
from plotnine import *
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ---------------------------------------------------- MAIN ---------------------------------------------------------- #
def main():
    # ======== Residual plot for linea regression model ======== #
    print("Loading and preparing data for linear regression model plots")
    regression = pd.read_csv("../dat/predictions_regression_four_features.csv")
    regression['Residuals'] = regression['y'] - regression['y_pred']

    p1 = (ggplot(data=regression) +
          aes(x='y_pred', y='Residuals', color='danceability') +
          geom_point() + labs(color='danceability'))
    warnings.filterwarnings("ignore", category=UserWarning)

    p1.save("../fig/regression_residuals.pdf", width=3, height=2.5)
    print("     ...saved residual plot under '../fig/regression_residuals.pdf'...")

    # ======== Histogram containing predicted and true popularity ======== #
    p2 = (ggplot(data=pd.melt(regression, value_vars=['y_pred', 'y'])) +
          aes(x='value', fill='variable') +
          geom_density(alpha=0.7) + labs(x='popularity'))

    p2.save("../fig/regression_true_vs_predicted.pdf", width=3, height=2.5)
    print("     ...saved histogram of predicted vs. true popularity under '../fig/regression_true_vs_predicted.pdf'...")

    # ======== Confusion matrix for logistic regression model ======== #
    print("Loading and preparing data for logistic regression confusion matrix")
    log_regression = pd.read_csv("../dat/predictions_log_regression_four_features.csv")

    cm_1 = confusion_matrix(log_regression['y'], log_regression['y_pred'])

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ConfusionMatrixDisplay(cm_1).plot(ax=ax)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.xticks(ticks=[0, 1], labels=["≤ 50", "> 50"])
    plt.yticks(ticks=[0, 1], labels=["≤ 50", "> 50"]);

    plt.savefig("../fig/log_regression_confusion.pdf")
    print("     ...saved confusion matrix under '../fig/log_regression_confusion.pdf'...")


if __name__ == "__main__":
    main()
