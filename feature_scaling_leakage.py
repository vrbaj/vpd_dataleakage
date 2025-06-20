"""
Quantification of feature scaling leakage in voice pathology detection using SVD.
"""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (MinMaxScaler, StandardScaler, MaxAbsScaler,
                                   RobustScaler, QuantileTransformer)
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


if __name__ == "__main__":
    RANDOM_STATE = 42
    EXPERIMENTS = 1000  # number of experiments
    INFORMATION_PRINT = False  # if true, printing results of each experiment
    LEAKAGE = [True, False]

    # load data
    dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))
    y = dataset["pathology"]
    # drop target and session_id columns
    X_orig = dataset.drop(columns=["pathology", "session_id"])

    # grid search params definitions for various classifiers
    PARAM_GRID_ADABOOST = {"classifier__n_estimators": [100, 150, 200, 250],
                           "classifier__learning_rate": [0.1, 1, 10]}
    PARAM_GRID_SVM = {"classifier__C": [1, 5, 10, 50, 100, 500, 1000, 5000],
                      "classifier__kernel": ["rbf"],
                      "classifier__gamma": [1.0, 0.5, 0.1, 0.05, 0.01, "auto"]}
    PARAM_GRID_RF = {"classifier__n_estimators": [50, 75, 100, 125, 150, 175, 200],
                     "classifier__criterion": ["gini"],
                     "classifier__min_samples_split": [2, 3, 4],
                     "classifier__max_features": ["sqrt"]}
    PARAM_GRID_DT = {"classifier__criterion": ["gini", "log_loss", "entropy"],
                     "classifier__min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                     "classifier__splitter": ["best", "random"],
                     "classifier__max_features": ["log2", "sqrt"]}
    PARAM_GRID_MLP = {
        'classifier__activation': ['relu'],
        'classifier__hidden_layer_sizes': [
            [int(2 * X_orig.shape[1])], [int(1.6 * X_orig.shape[1])],
            [int(1.2 * X_orig.shape[1])], [int(0.8 * X_orig.shape[1])]
        ]}
    PARAM_GRID_NB = {'classifier__var_smoothing': [1e-9]}
    PARAM_GRID_KNN = {'classifier__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
                      'classifier__weights': ['uniform', 'distance']}
    PARAM_GRID_LDA = {'classifier__solver': ['svd']}
    PARAM_GRID_QDA = {'classifier__tol': [0.0001],
                      'classifier__reg_param': [0.0, 0.0001, 0.01]}

    # classifiers with params grid
    clfs = {
        "gaussianNB": [PARAM_GRID_NB, GaussianNB()],
        "knn": [PARAM_GRID_KNN, KNeighborsClassifier()],
        "lda": [PARAM_GRID_LDA, LinearDiscriminantAnalysis()],
        'qda': [PARAM_GRID_QDA, QuadraticDiscriminantAnalysis()],
        "adaboost": [PARAM_GRID_ADABOOST, AdaBoostClassifier(random_state=RANDOM_STATE)],
        "svm": [PARAM_GRID_SVM, SVC(max_iter=int(5e5), random_state=RANDOM_STATE)],
        "rf": [PARAM_GRID_RF, RandomForestClassifier(random_state=RANDOM_STATE)],
        "dt": [PARAM_GRID_DT, DecisionTreeClassifier(random_state=RANDOM_STATE)],
        "mlp": [PARAM_GRID_MLP, MLPClassifier(max_iter=10000, random_state=RANDOM_STATE,
                                              solver="lbfgs")]
    }

    for clf_name, clf_settings in clfs.items():
        # get params for gridsearch
        clf_grid = clf_settings[0]
        # get classifier
        clf = clf_settings[1]
        results = {}
        for scaler in [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), RobustScaler(),
            QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)]:
            print(f"Running {clf_name} with {scaler.__class__.__name__}")
            for split_seed in tqdm(range(EXPERIMENTS)):
                np.random.seed(split_seed)
                # matthews correlation coefficient
                mcc = {"leakage": 0.0,
                       "correct": 0.0}
                # balanced accuracy
                bac = {"leakage": 0.0,
                       "correct": 0.0}

                for leakage in LEAKAGE:
                    X = deepcopy(X_orig)
                    if leakage:
                        # introducing data leakage by applying transformation on the whole dataset
                        X = scaler.fit_transform(X)
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=split_seed)
                        pipeline = Pipeline(steps=[("classifier", clf)])
                    else:
                        # correct approach, fit transformation on training data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=split_seed)
                        pipeline = Pipeline(steps=[("scaler", scaler), ("classifier", clf)])

                    grid_search = GridSearchCV(pipeline, param_grid=clf_grid, cv=5, n_jobs=-1)
                    grid_search.fit(X_train, y_train)
                    # Evaluate the best model on the test set
                    y_pred = grid_search.predict(X_test)

                    # Compute Matthews correlation coefficient
                    mcc_test = matthews_corrcoef(y_test, y_pred)

                    # Compute balanced accuracy score
                    bac_test = balanced_accuracy_score(y_test, y_pred)
                    # Save results to dictionary
                    if leakage:
                        mcc["leakage"] = mcc_test
                        bac["leakage"] = bac_test
                    else:
                        mcc["correct"] = mcc_test
                        bac["correct"] = bac_test
                results[split_seed] = {"mcc": mcc, "bcc": bac}
                if INFORMATION_PRINT:
                    print(f"Difference {mcc["leakage"] - mcc["correct"]} in Matthews "
                          f"Correlation Coefficient for seed {split_seed}, "
                          f"mcc correct: {mcc['correct']}")
                    print(f"Difference {bac["leakage"] - bac["correct"]} in Balanced"
                          f" Accuracy for seed {split_seed}, "
                          f"bcc correct: {bac['correct']}")
            # dump results
            filename_to_save = (f"results_{scaler.__class__.__name__}"
                                f"_leakage_{EXPERIMENTS}_{clf_name}.json")
            # create results folder if needed
            Path("results").mkdir(exist_ok=True)
            with open(Path("results", filename_to_save), "w",
                      encoding="utf8") as results_file:
                json.dump(results, results_file)
