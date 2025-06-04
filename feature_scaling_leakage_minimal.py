"""
Quantification of feature scaling leakage in voice pathology detection using SVD.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


if __name__ == "__main__":
    RANDOM_STATE = 42
    EXPERIMENTS = 1  # number of experiments
    INFORMATION_PRINT = False  # if true, printing results of each experiment
    LEAKAGE = [True, False]

    # load data
    dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))
    y = dataset["pathology"]
    # drop target and session_id columns
    X = dataset.drop(columns=["pathology", "session_id"])

    # grid definitions
    PARAM_GRID_ADABOOST = {"n_estimators": 300,
                           "learning_rate": 0.1,
                           "random_state": RANDOM_STATE}
    PARAM_GRID_SVM = {"C": 500,
                      "kernel": "rbf",
                      "gamma": "auto",
                      "random_state": RANDOM_STATE,
                      "max_iter": 100000}
    PARAM_GRID_RF = {"n_estimators": 175,
                     "criterion": "gini",
                     "min_samples_split": 4,
                     "max_features": "sqrt",
                     "random_state": RANDOM_STATE}
    PARAM_GRID_DT = {"criterion": "gini",
                     "min_samples_split": 10,
                     "splitter": "random",
                     "max_features": "sqrt",
                     "random_state": RANDOM_STATE}
    PARAM_GRID_MLP = {
        'activation': 'relu',
        'hidden_layer_sizes': [int(2 * X.shape[1])],
        "random_state": RANDOM_STATE,
        "solver": "lbfgs",
        "max_iter": 10000,
    }
    PARAM_GRID_NB = {'var_smoothing': 1e-9}
    PARAM_GRID_KNN = {'n_neighbors': 11,
                      'weights': 'distance'}
    PARAM_GRID_LDA = {'solver': 'svd'}
    PARAM_GRID_QDA = {'tol': 0.0001,
                      'reg_param': 0.01}
    PARAM_GRID_GP = {'kernel': RBF(),
                     "random_state": RANDOM_STATE}



    # classifiers with params grid
    clfs = {
        "gaussianNB": GaussianNB(**PARAM_GRID_NB),
        "knn": KNeighborsClassifier(**PARAM_GRID_KNN),
        "lda": LinearDiscriminantAnalysis(**PARAM_GRID_LDA),
        'qda': QuadraticDiscriminantAnalysis(**PARAM_GRID_QDA),
        'gaussian_process': GaussianProcessClassifier(**PARAM_GRID_GP),
        "adaboost": AdaBoostClassifier(**PARAM_GRID_ADABOOST),
        "svm": SVC(**PARAM_GRID_SVM),
        "rf": RandomForestClassifier(**PARAM_GRID_RF),
        "dt": DecisionTreeClassifier(**PARAM_GRID_DT),
        "mlp": MLPClassifier(**PARAM_GRID_MLP)
    }

    for clf_name, clf in clfs.items():

        results = {}
        for scaler in [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), RobustScaler(),
            QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)]:
            print(f"Running {clf_name} with {scaler.__class__.__name__}")
            for split_seed in tqdm(range(EXPERIMENTS)):
                np.random.seed(split_seed)
                # matthews correlation coefficient
                mcc = {"leakage": 0,
                       "correct": 0}
                # balanced accuracy
                bcc = {"leakage": 0,
                       "correct": 0}

                for leakage in LEAKAGE:
                    if leakage:
                        # introducing data leakage by applying transformation on the whole dataset
                        X = scaler.fit_transform(X)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=split_seed)


                        clf.fit(X_train, y_train)
                    else:
                        # correct approach, fit transformation on training data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=split_seed)


                        clf.fit(X_train, y_train)
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)



                    # Evaluate the best model on the test set

                    y_pred = clf.predict(X_test)

                    # Compute and display Matthews correlation coefficient
                    mcc_test = matthews_corrcoef(y_test, y_pred)

                    # Compute balance accuracy score
                    bac_test = balanced_accuracy_score(y_test, y_pred)

                    if leakage:
                        mcc["leakage"] = mcc_test
                        bcc["leakage"] = bac_test
                    else:
                        mcc["correct"] = mcc_test
                        bcc["correct"] = bac_test
                results[split_seed] = {"mcc": mcc, "bcc": bcc}
                if INFORMATION_PRINT:
                    print(f"Difference {mcc["leakage"] - mcc["correct"]} in Matthews "
                          f"Correlation Coefficient for seed {split_seed}, "
                          f"mcc correct: {mcc['correct']}")
                    print(f"Difference {bcc["leakage"] - bcc["correct"]} in Balanced"
                          f" Accuracy for seed {split_seed}, "
                          f"bcc correct: {bcc['correct']}")
            # dump results
            filename_to_save = (f"results_{scaler.__class__.__name__}"
                                f"_leakage_{EXPERIMENTS}_{clf_name}.json")
            with open(Path("results_minimal", filename_to_save), "w",
                      encoding="utf8") as results_file:
                json.dump(results, results_file)
