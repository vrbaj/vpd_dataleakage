import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, matthews_corrcoef
from tqdm import tqdm


dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))

leakage_param = [True, False]


param_grid = {"n_estimators": [50, 100, 150, 200, 250, 300, 350, 400],
              "learning_rate": [0.1, 1, 10]}
results = {}
for SEED_N in tqdm(range(1)):
    mcc = {"leakage": 0,
           "correct": 0}
    for leakage in leakage_param:
        y = dataset["pathology"]
        X = dataset.drop(columns=["pathology", "session_id"])
        scaler = MinMaxScaler()
        if leakage:
            X = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=SEED_N)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                random_state=SEED_N)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


        adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=SEED_N)

        grid_search = GridSearchCV(estimator=adaboost, param_grid=param_grid, scoring="matthews_corrcoef",
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
    results[SEED_N] = mcc
    # if mcc_diff != 0:
    #     print(f"Difference {mcc["leakage"] - mcc["correct"]} in Matthews Correlation Coefficient for seed {SEED_N}:")
    #     break
with open("results_scaling_leakage.json", "w") as f:
    json.dump(results, f)
