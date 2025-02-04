"""
Quantification of feature scaling leakage in voice pathology detection using SVD.
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm


if __name__ == "__main__":
    EXPERIMENTS = 200  # number of experiments
    INFORMATION_PRINT = True  # if true, printing results of each experiment
    LEAKAGE = [True, False]
    # grid definitions
    PARAM_GRID_ADABOOST = {"n_estimators": [200, 250, 300, 350, 400],
                           "learning_rate": [0.1, 1, 10]}
    PARAM_GRID_SVM = {"C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
                      "kernel": ["rbf"],
                      "gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"], }
    PARAM_GRID_RF = {"n_estimators": [50, 75, 100, 125, 150, 175],
                     "criterion": ["gini"],
                     "min_samples_split": [2, 3, 4, 5, 6],
                     "max_features": ["sqrt"]}
    PARAM_GRID_DT = {"criterion": ["gini", "log_loss", "entropy"],
                     "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
                     "splitter": ["best", "random"],
                     "max_features": ["log2", "sqrt"]}
    PARAM_GRID_MLP = {
        'activation': ['relu', 'tanh'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'alpha': [0.0001, 0.001, 0.01]
    }

    # load data
    dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))
    y = dataset["pathology"]
    # drop target and session_id columns
    X = dataset.drop(columns=["pathology", "session_id"])

    # classifiers with params grid
    clfs = {
        "adaboost": [PARAM_GRID_ADABOOST, AdaBoostClassifier(algorithm='SAMME', random_state=42)],
        "svm": [PARAM_GRID_SVM, SVC(random_state=42, max_iter=int(5e5))],
        "rf": [PARAM_GRID_RF, RandomForestClassifier(random_state=42)],
        "dt": [PARAM_GRID_DT, DecisionTreeClassifier(random_state=42)],
        "mlp": [PARAM_GRID_MLP, MLPClassifier(max_iter=10000, random_state=42,
                                              hidden_layer_sizes=int(0.8 * X.shape[1]))],
    }

    for clf_name, clf_settings in clfs.items():
        clf_grid = clf_settings[0]
        clf = clf_settings[1]
        results = {}
        for scaler in [MinMaxScaler(), StandardScaler()]:
            print(f"Running {clf_name} with {scaler.__class__.__name__}")
            for split_seed in tqdm(range(EXPERIMENTS)):
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
                            X, y, test_size=0.2, random_state=split_seed, stratify=y)
                    else:
                        # correct approach, fit transformation on training data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=0.2, random_state=split_seed, stratify=y)
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    # perform grid search for given classifier and corresponding grid
                    grid_search = GridSearchCV(estimator=clf, param_grid=clf_grid,
                                               scoring="balanced_accuracy",
                                               cv=10, n_jobs=-1)

                    # Fit the grid search to the training data
                    grid_search.fit(X_train, y_train)

                    # Evaluate the best model on the test set
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)

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
            with open(Path("results", filename_to_save), "w",
                      encoding="utf8") as results_file:
                json.dump(results, results_file)
