from pathlib import Path
import pandas as pd
import ast

# Load the CSV file
df = pd.read_csv(Path("data", "features.csv"))

# Function to parse stringified lists
def parse_features(features):
    try:
        # Convert stringified list to actual Python list
        return ast.literal_eval(features) if isinstance(features, str) else features
    except (ValueError, SyntaxError):
        return []

# Apply the function to parse the features column
features_to_flatten = ["spectral_contrast", "formants", "mfcc", "var_mfcc", "delta_mfcc", "var_delta_mfcc",
                       "delta2_mfcc", "var_delta2_mfcc", "lfcc"]
for feature in features_to_flatten:
    df[feature] = df[feature].apply(parse_features)

    # Find the maximum length of the lists
    max_length = df[feature].apply(len).max()

    # Expand the lists into separate columns
    for i in range(max_length):
        df[f'{feature}_{i+1}'] = df[feature].apply(lambda x: x[i] if i < len(x) else None)

    # Drop the original features column
    df = df.drop(columns=[feature])

# Save the new CSV
df.to_csv(Path("data", "flattened_features.csv"), index=False)
