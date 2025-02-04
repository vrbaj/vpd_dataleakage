"""
Script to demonstrate the importance of test set.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


if __name__ == "__main__":
    # params for SVM
    C_PARAMS = [0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 3000, 5000, 7000, 10000, 12000]
    GAMMAS = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, "auto"]

    # load data
    dataset = pd.read_csv(Path(".", "data", "flattened_features.csv"))
    y = dataset["pathology"]
    # drop target and session_id columns
    X = dataset.drop(columns=["pathology", "session_id"])
    bac_results = []
    for c_param in tqdm(C_PARAMS):
        for gamma_param in GAMMAS:
            clf = SVC(random_state=42, max_iter=int(5e5), C=c_param,
                      gamma=gamma_param, kernel="rbf")
            clf.fit(X, y)
            bac_results.append(balanced_accuracy_score(y, clf.predict(X)))
    print(f"Total number of experiments: {len(bac_results)}")
    print(f"Number of 100% accuracy results: "
          f"{len(np.where(np.asarray(bac_results) == 1.0)[0])}")
