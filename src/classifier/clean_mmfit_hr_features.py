from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_DIR = Path(r"C:\Masters\data\features\mmfit")

INPUT_PATH = FEATURE_DIR / "mmfit_fusion_features_with_hr.csv"
OUT_PATH = FEATURE_DIR / "mmfit_fusion_features_with_hr_cleaned.csv"

data = pd.read_csv(INPUT_PATH)

# -------------------------------------------------------
# Helper: combine left and right HR safely
# -------------------------------------------------------

def combine_hr(row, feature_name):
    left_col = f"sw_l_hr_{feature_name}"
    right_col = f"sw_r_hr_{feature_name}"

    left = row[left_col] if left_col in row else np.nan
    right = row[right_col] if right_col in row else np.nan

    left_valid = pd.notna(left)
    right_valid = pd.notna(right)

    if left_valid and right_valid:
        return np.mean([left, right])
    elif left_valid:
        return left
    elif right_valid:
        return right
    else:
        return np.nan


# -------------------------------------------------------
# Create combined HR features
# -------------------------------------------------------

hr_features = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "change",
    "slope",
]

for feature in hr_features:
    data[f"hr_{feature}_combined"] = data.apply(
        lambda row: combine_hr(row, feature),
        axis=1
    )


# -------------------------------------------------------
# Create HR availability flags
# -------------------------------------------------------

data["hr_left_available"] = data["sw_l_hr_mean"].notna().astype(int)
data["hr_right_available"] = data["sw_r_hr_mean"].notna().astype(int)

data["hr_any_available"] = (
    (data["hr_left_available"] == 1) |
    (data["hr_right_available"] == 1)
).astype(int)

data["hr_both_available"] = (
    (data["hr_left_available"] == 1) &
    (data["hr_right_available"] == 1)
).astype(int)


# -------------------------------------------------------
# Check left/right disagreement
# -------------------------------------------------------

data["hr_left_right_diff"] = (
    data["sw_l_hr_mean"] - data["sw_r_hr_mean"]
).abs()

# Big disagreement threshold.
# If left and right mean HR differ by more than 20 bpm, flag it.
data["hr_disagreement_flag"] = (
    (data["hr_both_available"] == 1) &
    (data["hr_left_right_diff"] > 20)
).astype(int)


# -------------------------------------------------------
# Create safer HR feature for proxy labels
# -------------------------------------------------------

# If disagreement is high, HR should be treated as lower-confidence.
# We still keep the combined value, but we mark the issue.
data["hr_quality_flag"] = "no_hr"

data.loc[
    (data["hr_any_available"] == 1) & (data["hr_disagreement_flag"] == 0),
    "hr_quality_flag"
] = "usable"

data.loc[
    (data["hr_any_available"] == 1) & (data["hr_disagreement_flag"] == 1),
    "hr_quality_flag"
] = "left_right_disagreement"


# -------------------------------------------------------
# Save cleaned fusion file
# -------------------------------------------------------

data.to_csv(OUT_PATH, index=False)

print("Saved:")
print(OUT_PATH)

print("\nShape:")
print(data.shape)

print("\nHR availability:")
print(data[[
    "hr_left_available",
    "hr_right_available",
    "hr_any_available",
    "hr_both_available",
    "hr_disagreement_flag"
]].sum())

print("\nHR quality counts:")
print(data["hr_quality_flag"].value_counts())

print("\nCombined HR summary:")
print(data[[
    "hr_mean_combined",
    "hr_min_combined",
    "hr_max_combined",
    "hr_change_combined",
    "hr_slope_combined",
    "hr_left_right_diff"
]].describe())

print("\nPreview:")
preview_cols = [
    "participant",
    "start_time",
    "end_time",
    "exercise",
    "sw_l_hr_mean",
    "sw_r_hr_mean",
    "hr_mean_combined",
    "hr_left_right_diff",
    "hr_quality_flag",
]

print(data[preview_cols].head(30))