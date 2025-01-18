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
from sklearn.metrics import matthews_corrcoef

from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


results = {}
RANGE_START = 0
RANGE_END = 200

dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))
y = dataset["pathology"]
X = dataset.drop(columns=["pathology", "session_id"])

leakage_param = [True, False]


param_grid_adaboost = {"n_estimators": [200, 250, 300, 350, 400],
              "learning_rate": [0.1, 1, 10]}
param_grid_svm = { "C": [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000],
        "kernel": ["rbf"],
        "gamma": [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"],}
param_grid_rf = {"n_estimators": [50, 75, 100, 125, 150, 175],
        "criterion": ["gini"],
        "min_samples_split": [2, 3, 4, 5, 6],
        "max_features": ["sqrt"]}
param_grid_dt ={"criterion": ["gini", "log_loss", "entropy"],
        "min_samples_split": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "splitter": ["best", "random"],
        "max_features": ["log2", "sqrt"]}
param_grid_mlp = {
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha': [0.0001, 0.001, 0.01]
}


clfs = {
    "adaboost": [param_grid_adaboost, AdaBoostClassifier(algorithm='SAMME', random_state=42)],
    "svm": [param_grid_svm, SVC(random_state=42, max_iter=int(5e5))],
    "rf": [param_grid_rf, RandomForestClassifier(random_state=42)],
    "dt": [param_grid_dt, DecisionTreeClassifier(random_state=42)],
    "mlp": [param_grid_mlp, MLPClassifier(max_iter=10000, random_state=42,
                                    hidden_layer_sizes=int(0.8 * X.shape[1]))],
}

for clf_name, clf_settings in clfs.items():
    clf_grid = clf_settings[0]
    clf = clf_settings[1]
    for scaler in [MinMaxScaler(), StandardScaler()]:
        for split_seed in range(RANGE_START, RANGE_END):
            mcc = {"leakage": 0,
                   "correct": 0}
            for leakage in leakage_param:
                if leakage:
                    X = scaler.fit_transform(X)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                        random_state=split_seed, stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                        random_state=split_seed, stratify=y)
                    X_train = scaler.fit_transform(X_train)
                    X_test = scaler.transform(X_test)


                # adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=42)
                # svm_clf = SVC(random_state=42, max_iter=int(5e5))
                # rf_clf = RandomForestClassifier(random_state=42)
                # dt_clf = DecisionTreeClassifier(random_state=42)
                # mlp_clf = MLPClassifier(max_iter=10000, random_state=42,
                #                         hidden_layer_sizes=(int(0.8 * X_train.shape[1])))
                grid_search = GridSearchCV(estimator=clf, param_grid=clf_grid, scoring="matthews_corrcoef",
                                           cv=10, n_jobs=-1)

                # Fit the grid search to the training data
                grid_search.fit(X_train, y_train)

                # Print the best parameters and best score
                # print("Best Parameters:", grid_search.best_params_)
                # print("Best Cross-Validation Score:", grid_search.best_score_)

                # Evaluate the best model on the test set
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)


                # Compute and display Matthews correlation coefficient
                mcc_test = matthews_corrcoef(y_test, y_pred)
                if leakage:
                    mcc["leakage"] = mcc_test
                else:
                    mcc["correct"] = mcc_test
            mcc_diff = mcc["leakage"] - mcc["correct"]
            results[split_seed] = mcc
            print(f"Difference {mcc["leakage"] - mcc["correct"]} in Matthews Correlation Coefficient for seed {split_seed}, "
                  f"mcc correct: {mcc['correct']}")

        with open(f"results_{scaler.__class__.__name__}_leakage_{RANGE_START}_{RANGE_END}_{clf_name}.json",
                  "w", encoding="utf8") as f:
            json.dump(results, f)
