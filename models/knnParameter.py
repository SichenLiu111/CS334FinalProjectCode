import argparse
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import argparse


def main():
    """
    Main file to run from the command line.
    set up the program to take in arguments from the command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--xTrain",
                        default="xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="yTrain_discrete.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="yTest_discrete.csv",
                        help="filename for labels associated with the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)


    #  Define the parameter grid
    param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
    }

    # Initialize a KNN classifier
    knn = KNeighborsClassifier()

    # Setup the grid search
    grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=1, n_jobs=-1)

    # Fit the grid search to your data
    grid_search.fit(xTrain, yTrain)

    # Best parameter set
    print("Best parameters found: ", grid_search.best_params_)
    print("Best model found: ", grid_search.best_estimator_)

    
if __name__ == "__main__":
    main()