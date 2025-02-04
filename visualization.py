import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import shapiro, kstest, normaltest

METRIC_OF_INTEREST = "bcc"
CLASSIFIER_RESULTS = "svm"
SCALER = "MinMaxScaler"

RESULTS_FILE_NAME = f"results_{SCALER}_leakage_0_200_{CLASSIFIER_RESULTS}.json"
def check_normality(data):
    print("\n--- Statistical Tests ---\n")

    # Shapiro-Wilk Test
    stat_shapiro, p_shapiro = shapiro(data)
    print(f"Shapiro-Wilk Test: Statistic={stat_shapiro:.3f}, p-value={p_shapiro:.3f}")

    # D'Agostino and Pearson's Test
    stat_normaltest, p_normaltest = normaltest(data)
    print(f"D'Agostino and Pearson's Test: Statistic={stat_normaltest:.3f}, p-value={p_normaltest:.3f}")

    # Interpretation
    if p_shapiro > 0.05 and p_normaltest > 0.05:
        print("Data appears to be normally distributed (fail to reject H0).")
    else:
        print("Data does not appear to be normally distributed (reject H0).")


with open(Path("results", RESULTS_FILE_NAME), "r") as f:
    data_l = json.load(f)
mcc_differences = []

for item in data_l.values():

    if item[METRIC_OF_INTEREST]["leakage"] != 0 and item[METRIC_OF_INTEREST]["correct"] != 0:
        mcc_differences.append(item[METRIC_OF_INTEREST]["leakage"] - item[METRIC_OF_INTEREST]["correct"])

check_normality(np.array(mcc_differences))
print(f"average difference in {METRIC_OF_INTEREST} is: {np.average(np.array(mcc_differences))}" )
print(f"median is {np.median(np.array(mcc_differences)) }")
print(f"max: {np.max(np.array(mcc_differences))}  min: {np.min(np.array(mcc_differences)) }")
plt.hist(mcc_differences, bins=50)
plt.xlabel("MCC difference (leakage - correct)")
plt.ylabel("Count")
plt.title(f"{CLASSIFIER_RESULTS} - {SCALER}")
plt.text(-0.08, 125, f"Median = {np.median(mcc_differences):0.4f}")
plt.text(-0.08, 115, f"Min = {np.min(mcc_differences):0.4f}")
plt.text(-0.08, 105, f"Max = {np.max(mcc_differences):0.4f}")
plt.text(-0.08, 95, f"Only two experiments < 0.")
plt.show()
