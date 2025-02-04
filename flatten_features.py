"""
Script that flattens lists in features.csv to flattened_features.csv
"""
import ast
from pathlib import Path

import pandas as pd


def parse_features(features):
    """
    Transform string to value or list of values.
    :param features: string representing column in features.csv
    :return: list or number
    """
    # Function to parse stringified lists
    try:
        # Convert stringified list to actual Python list
        return ast.literal_eval(features) if isinstance(features, str) else features
    except (ValueError, SyntaxError):
        return []


if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv(Path("data", "features.csv"))
    # Apply the function to parse the features column
    features_to_flatten = ["spectral_contrast", "formants", "mfcc", "var_mfcc",
                           "delta_mfcc", "var_delta_mfcc",
                           "delta2_mfcc", "var_delta2_mfcc", "lfcc"]
    for feature in features_to_flatten:
        df[feature] = df[feature].apply(parse_features)
        # Find the maximum length of the lists
        max_length = df[feature].apply(len).max()
        # Expand the lists into separate columns
        for idx in range(max_length):
            # pylint:disable=cell-var-from-loop
            df[f"{feature}_{idx + 1}"] = df[feature].apply(
                lambda x: x[idx] if idx < len(x) else None)
        # Drop the original features column
        df = df.drop(columns=[feature])

    # Save the new CSV
    df.to_csv(Path("data", "flattened_features.csv"), index=False)
