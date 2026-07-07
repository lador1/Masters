from pathlib import Path
import pandas as pd

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")

POSE_PATH = FEATURE_DIR / "mmfit_pose_features_v2_fixed_clean.csv"
WEARABLE_PATH = FEATURE_DIR / "mmfit_wearable_features_basic.csv"
OUT_PATH = FEATURE_DIR / "mmfit_fusion_features_basic.csv"

pose = pd.read_csv(POSE_PATH)
wearable = pd.read_csv(WEARABLE_PATH)

print("Pose shape:")
print(pose.shape)

print("\nWearable shape:")
print(wearable.shape)

merge_cols = [
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

fusion = pd.merge(
    pose,
    wearable,
    on=merge_cols,
    how="inner",
    suffixes=("_pose", "_wearable")
)

fusion.to_csv(OUT_PATH, index=False)

print("\nFusion feature table saved to:")
print(OUT_PATH)

print("\nFusion shape:")
print(fusion.shape)

print("\nExercise counts:")
print(fusion["exercise"].value_counts())

print("\nMissing values:")
print(fusion.isna().sum().sort_values(ascending=False).head(30))

print("\nNumber of columns:")
print(len(fusion.columns))