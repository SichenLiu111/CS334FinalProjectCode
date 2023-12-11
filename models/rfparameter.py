from sklearn.ensemble import RandomForestClassifier
import argparse
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import GridSearchCV


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

    param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

    # Initialize a Random Forest classifier
    rf = RandomForestClassifier()

    # Setup the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit the grid search to your data
    grid_search.fit(xTrain, yTrain)

    # Best parameter set
    print("Best parameters found: ", grid_search.best_params_)
    print("Best model found: ", grid_search.best_estimator_)



if __name__ == "__main__":
    main()


# Best parameters found:  {'bootstrap': True, 'max_depth': 4, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}
# Best model found:  RandomForestClassifier(max_depth=4, min_samples_leaf=4, min_samples_split=10)
