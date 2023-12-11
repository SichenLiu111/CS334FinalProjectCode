from sklearn.metrics import mean_squared_error
import argparse
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report


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


    # Initialize the Random Forest classifier with the specified parameters
    rf_model = RandomForestClassifier(
        bootstrap=True, 
        max_depth=4, 
        min_samples_leaf=4, 
        min_samples_split=10, 
        n_estimators=100
    )

    # Train the model on your training data
    rf_model.fit(xTrain, yTrain)

    # Make predictions
    predictions = rf_model.predict(xTest)

    # Optionally, evaluate the model
    accuracy = accuracy_score(yTest, predictions)
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(yTest, predictions))





# OOB Score: 0.31834126303647714
# Mean Squared Error on Test Set: 0.17114506270027569

if __name__ == "__main__":
    main()
