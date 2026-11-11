from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\features\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "mmfit_heart_rate_features_basic.csv"

rows = []


def safe_load(path):
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def extract_hr_window(hr_array, start, end):
    """
    MM-Fit HR arrays appear to have 3 columns:
    column 0 = MM-Fit frame/time index
    column 1 = timestamp
    column 2 = heart-rate value

    We select rows where column 0 is within the exercise window.
    Then we use column 2 as the actual heart-rate value.
    """
    if hr_array is None:
        return np.array([])

    hr_array = np.asarray(hr_array, dtype=float)

    if hr_array.ndim != 2 or hr_array.shape[1] < 3:
        return np.array([])

    frame_col = hr_array[:, 0]
    hr_col = hr_array[:, 2]

    mask = (frame_col >= start) & (frame_col <= end)
    values = hr_col[mask]

    values = values[np.isfinite(values)]

    # remove impossible HR values
    values = values[(values >= 30) & (values <= 220)]

    return values


def hr_features(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_valid": 0,
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_change": np.nan,
            f"{prefix}_slope": np.nan,
        }

    x = np.arange(len(values))

    if len(values) > 1:
        slope = np.polyfit(x, values, 1)[0]
        change = values[-1] - values[0]
    else:
        slope = 0.0
        change = 0.0

    return {
        f"{prefix}_valid": 1,
        f"{prefix}_n": len(values),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_change": float(change),
        f"{prefix}_slope": float(slope),
    }


participants = sorted([
    p for p in RAW_DIR.iterdir()
    if p.is_dir() and p.name.startswith("w")
])

for p_dir in participants:
    participant = p_dir.name
    label_path = p_dir / f"{participant}_labels.csv"

    if not label_path.exists():
        continue

    labels = pd.read_csv(
        label_path,
        header=None,
        names=["start_time", "end_time", "reps", "exercise"]
    )

    left_hr = safe_load(p_dir / f"{participant}_sw_l_hr.npy")
    right_hr = safe_load(p_dir / f"{participant}_sw_r_hr.npy")

    for _, label_row in labels.iterrows():
        start = int(label_row["start_time"])
        end = int(label_row["end_time"])

        left_values = extract_hr_window(left_hr, start, end)
        right_values = extract_hr_window(right_hr, start, end)

        output = {
            "participant": participant,
            "start_time": start,
            "end_time": end,
            "duration": end - start,
            "reps": label_row["reps"],
            "exercise": label_row["exercise"],
        }

        output.update(hr_features(left_values, "sw_l_hr"))
        output.update(hr_features(right_values, "sw_r_hr"))

        rows.append(output)

features = pd.DataFrame(rows)
features.to_csv(OUT_PATH, index=False)

print("Saved:")
print(OUT_PATH)
print("Shape:", features.shape)
print(features.head())

print("\nValid counts:")
print(features[["sw_l_hr_valid", "sw_r_hr_valid"]].sum())

print("\nHeart-rate summary:")
hr_cols = [
    "sw_l_hr_mean", "sw_l_hr_min", "sw_l_hr_max",
    "sw_r_hr_mean", "sw_r_hr_min", "sw_r_hr_max",
]
print(features[hr_cols].describe())