from pathlib import Path
import pandas as pd

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")

FUSION_PATH = FEATURE_DIR / "mmfit_fusion_features_basic.csv"
HR_PATH = FEATURE_DIR / "mmfit_heart_rate_features_basic.csv"
OUT_PATH = FEATURE_DIR / "mmfit_fusion_features_with_hr.csv"

fusion = pd.read_csv(FUSION_PATH)
hr = pd.read_csv(HR_PATH)

merge_cols = [
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
]

merged = fusion.merge(
    hr,
    on=merge_cols,
    how="left",
    validate="one_to_one"
)

merged.to_csv(OUT_PATH, index=False)

print("Saved:")
print(OUT_PATH)

print("\nFusion shape:")
print(fusion.shape)

print("\nHR shape:")
print(hr.shape)

print("\nMerged shape:")
print(merged.shape)

hr_cols = [c for c in merged.columns if "_hr_" in c]

print("\nHR columns added:")
for col in hr_cols:
    print("-", col)

print("\nMissing values in HR columns:")
print(merged[hr_cols].isna().sum())

print("\nValid HR segment counts:")
valid_cols = [c for c in ["sw_l_hr_valid", "sw_r_hr_valid"] if c in merged.columns]
print(merged[valid_cols].sum())

print("\nPreview:")
preview_cols = [
    "participant",
    "start_time",
    "end_time",
    "exercise",
    "sw_l_hr_mean",
    "sw_r_hr_mean",
    "sw_l_hr_slope",
    "sw_r_hr_slope",
]
preview_cols = [c for c in preview_cols if c in merged.columns]
print(merged[preview_cols].head(20))