import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    # Paths to the data files
    x_train_path = '/Users/zhenyanli/PycharmProjects/CS334/main/project/xTrain.csv'
    y_train_path = '/Users/zhenyanli/PycharmProjects/CS334/main/project/yTrain_discrete.csv'
    x_test_path = '/Users/zhenyanli/PycharmProjects/CS334/main/project/xTest.csv'
    y_test_path = '/Users/zhenyanli/PycharmProjects/CS334/main/project/yTest_discrete.csv'

    # Load the datasets
    x_train = pd.read_csv(x_train_path)
    y_train = pd.read_csv(y_train_path)
    x_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path)

    return x_train, y_train, x_test, y_test

def train_and_evaluate(x_train, y_train, x_test, y_test):
    # Create the Logistic Regression model
    logistic_model = LogisticRegression()

    # Define the hyperparameters to tune
    hyperparameters = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Norm of penalization
        'solver': ['liblinear']  # Solver that supports both l1 and l2 penalties
    }

    # Set up the grid search with cross-validation
    grid_search = GridSearchCV(logistic_model, hyperparameters, cv=5, verbose=0)

    # Perform the grid search and train the best model
    grid_search.fit(x_train, y_train.values.ravel())

    # Retrieve the best hyperparameters
    best_params = grid_search.best_params_
    print("Best hyperparameters:", best_params)

    # Make predictions with the best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)
    y_pred_proba = best_model.predict_proba(x_test)[:,1]

    # Perform the grid search and train the best model
    grid_search.fit(x_train, y_train.values.ravel())

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Calculate ROC Curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(report)
    return grid_search


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    for idx, val in enumerate(grid_param_2):
        plt.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    plt.title("Grid Search Scores")
    plt.xlabel(name_param_1)
    plt.ylabel('CV Average Score')
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_data()
    grid_search = train_and_evaluate(x_train, y_train, x_test, y_test)

    # Plot the results
    hyperparameters = grid_search.param_grid
    plot_grid_search(grid_search.cv_results_, hyperparameters['C'], hyperparameters['penalty'], 'C', 'Penalty')

if __name__ == "__main__":
    main()