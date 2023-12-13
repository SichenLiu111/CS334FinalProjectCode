# ETFs return prediction

This repository contains all code files for the projects. The following lines gives a structure of this repository.

preprocess: contains rawdata, code for preprocessing, and preprocessed datasets.

models: contains the code for hyperparameter tuning, training, testing, and evaluation for the five models used in this project.

draw_roc.ipynb: generate the ROC curves for the five models in one graph.

baseline_acc.ipynb: get the baseline accuracy for evaluation.

generate_raw.ipynb: read the raw dataset from the kaggle data folder. Before run this file, you will need to have the ETFs folder from the kaggle dataset and configure its path on your device correctly. You do not need to run this file youself. The rawdata has already been generated.

Note that to run those files, you will need to check the input dataset files paths. It is highly recommanded to direction use rawdata.csv for the preprocessing. The preprocessed datasets are at `preprocess/preprocessed_data'.


For the SVM and XGBoost models, the code was written and run in the google colab. The links to their implementations are shown in the following. You will need to upload data and check the file names and paths before you run the files.

SVM: https://colab.research.google.com/drive/1KXoCM9IDZdSB8RjeEH3W79cdudrSM9C8?usp=sharing

XGBoost: https://colab.research.google.com/drive/1Jb-2lsokr77bP_KhF-SqYA4IJhTqhmOT?usp=sharing


Contributions:

Sichen Liu: generate_raw.ipynb, preprocess.ipynb, stockETF_SVM.ipynb, stockETF_xgboost.ipynb, draw_roc.ipynb, and baseline_acc.ipynb.

Jenny Guo: knn.py, knnParameter.py, rf.py, rfparameter.py.

Zhanyan Li: logreg.py.
