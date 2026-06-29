from pathlib import Path
import pandas as pd

FEATURE_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_pose_features_v2_fixed.csv")
OUT_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_pose_features_v2_fixed_clean.csv")

features = pd.read_csv(FEATURE_PATH)

clean = features[
    features["pose_valid"] == 1
].dropna()

clean.to_csv(OUT_PATH, index=False)

print("Clean fixed pose feature table saved to:")
print(OUT_PATH)

print("\nOriginal shape:")
print(features.shape)

print("\nClean shape:")
print(clean.shape)

print("\nRows removed:")
print(len(features) - len(clean))

print("\nExercise counts after cleaning:")
print(clean["exercise"].value_counts())

print("\nParticipants after cleaning:")
print(clean["participant"].nunique())