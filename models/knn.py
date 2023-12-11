import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score



def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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
    # create an instance of the model
    
    # Model Evaluation
    knn = KNeighborsClassifier(args.k)
    knn.fit(xTrain, yTrain)

    # Predictions
    predictions = knn.predict(xTrain)

    trainAcc = accuracy_score(yTrain,predictions)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy_score(yHatTest, yTest)
    f1 = f1_score(yTest, yHatTest)
    
    print(f'F1 Score: {f1}')
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    
    y_scores = knn.predict_proba(xTest)[:, 1]  # Get probabilities for the positive class

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(yTest, y_scores)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - KNN')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    main()
    
# knn = 9
# F1 Score: 0.5886213047910296
# Training Acc: 0.6972925643048488
# Test Acc: 0.5521880851654067