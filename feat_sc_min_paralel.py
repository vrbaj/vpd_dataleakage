"""
Parallelized quantification of feature scaling leakage in voice pathology detection using SVD.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from time import time


def run_experiment(clf_name, clf, scaler, EXPERIMENTS, X_orig, y, RANDOM_STATE, LEAKAGE, INFORMATION_PRINT, stratify):
    results = {}
    for split_seed in range(EXPERIMENTS):
        np.random.seed(split_seed)
        mcc = {"leakage": 0, "correct": 0}
        bcc = {"leakage": 0, "correct": 0}

        for leakage in LEAKAGE:
            X = X_orig.copy()
            if leakage:
                X = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=split_seed,
                    stratify=y if stratify else None)
                clf.fit(X_train, y_train)
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=split_seed,
                    stratify=y if stratify else None)
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
                clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)
            mcc_score = matthews_corrcoef(y_test, y_pred)
            bcc_score = balanced_accuracy_score(y_test, y_pred)

            if leakage:
                mcc["leakage"] = mcc_score
                bcc["leakage"] = bcc_score
            else:
                mcc["correct"] = mcc_score
                bcc["correct"] = bcc_score

        results[split_seed] = {"mcc": mcc, "bcc": bcc}

        if INFORMATION_PRINT:
            print(f"{clf_name} | {scaler.__class__.__name__} | Seed {split_seed} | "
                  f"MCC diff: {mcc['leakage'] - mcc['correct']:.4f}, "
                  f"BCC diff: {bcc['leakage'] - bcc['correct']:.4f}")

    filename = f"results_{scaler.__class__.__name__}_leakage_{EXPERIMENTS}_{clf_name}.json"
    dirname = "results_minimal"
    if stratify:
        dirname += "_stratified"
    Path(dirname).mkdir(parents=True, exist_ok=True)
    with open(Path(dirname, filename), "w", encoding="utf8") as f:
        json.dump(results, f)


if __name__ == "__main__":
    RANDOM_STATE = 42
    EXPERIMENTS = 1000
    INFORMATION_PRINT = False
    LEAKAGE = [True, False]
    STRATIFY = [True, False]

    dataset = pd.read_csv(Path("data", "flattened_features.csv"))
    y = dataset["pathology"]
    X = dataset.drop(columns=["pathology", "session_id"])

    PARAM_GRID = {
        "gaussianNB": GaussianNB(var_smoothing=1e-9),
        "knn": KNeighborsClassifier(n_neighbors=11, weights='distance'),
        "lda": LinearDiscriminantAnalysis(solver='svd'),
        'qda': QuadraticDiscriminantAnalysis(tol=0.0001, reg_param=0.01),
        'gaussian_process': GaussianProcessClassifier(kernel=RBF(), random_state=RANDOM_STATE),
        "adaboost": AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=RANDOM_STATE),
        "svm": SVC(C=500, kernel="rbf", gamma="auto", random_state=RANDOM_STATE, max_iter=100000),
        "rf": RandomForestClassifier(n_estimators=175, criterion="gini", min_samples_split=4,
                                     max_features="sqrt", random_state=RANDOM_STATE),
        "dt": DecisionTreeClassifier(criterion="gini", min_samples_split=10, splitter="random",
                                     max_features="sqrt", random_state=RANDOM_STATE),
        "mlp": MLPClassifier(activation='relu', hidden_layer_sizes=[int(2 * X.shape[1])],
                              random_state=RANDOM_STATE, solver="lbfgs", max_iter=10000)
    }

    scalers = [MaxAbsScaler(), MinMaxScaler(), StandardScaler(), RobustScaler(),
               QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)]

    start = time()
    Parallel(n_jobs=os.cpu_count())(
        delayed(run_experiment)(clf_name, clf, scaler, EXPERIMENTS, X, y, RANDOM_STATE, LEAKAGE, INFORMATION_PRINT, stratify)
        for clf_name, clf in PARAM_GRID.items()
        for scaler in scalers
        for stratify in STRATIFY
    )
    print(f"Total duration of script: {time() - start}")
