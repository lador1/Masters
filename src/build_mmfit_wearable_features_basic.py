from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\features\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

label_files = sorted(RAW_DIR.rglob("*_labels.csv"))


def extract_sensor_features(sensor_path, start_time, end_time, prefix):
    """
    Extract basic smartwatch accelerometer/gyroscope features for one labelled exercise segment.

    MM-Fit wearable sensor files usually have shape:
    (samples, 5)

    Expected columns:
    column 0 = relative time
    column 1 = timestamp
    columns 2, 3, 4 = x, y, z sensor axes
    """

    empty_features = {
        f"{prefix}_valid": 0,
        f"{prefix}_count": 0,
        f"{prefix}_mag_mean": np.nan,
        f"{prefix}_mag_std": np.nan,
        f"{prefix}_mag_min": np.nan,
        f"{prefix}_mag_max": np.nan,
        f"{prefix}_x_mean": np.nan,
        f"{prefix}_y_mean": np.nan,
        f"{prefix}_z_mean": np.nan,
        f"{prefix}_x_std": np.nan,
        f"{prefix}_y_std": np.nan,
        f"{prefix}_z_std": np.nan,
    }

    if not sensor_path.exists():
        return empty_features

    data = np.load(sensor_path, allow_pickle=True)

    if data.ndim != 2 or data.shape[1] < 5:
        return empty_features

    time_col = data[:, 0]
    xyz = data[:, 2:5]

    segment = xyz[(time_col >= start_time) & (time_col <= end_time)]

    if len(segment) < 2:
        return empty_features

    magnitude = np.sqrt(np.sum(segment ** 2, axis=1))

    return {
        f"{prefix}_valid": 1,
        f"{prefix}_count": int(len(segment)),
        f"{prefix}_mag_mean": float(np.nanmean(magnitude)),
        f"{prefix}_mag_std": float(np.nanstd(magnitude)),
        f"{prefix}_mag_min": float(np.nanmin(magnitude)),
        f"{prefix}_mag_max": float(np.nanmax(magnitude)),
        f"{prefix}_x_mean": float(np.nanmean(segment[:, 0])),
        f"{prefix}_y_mean": float(np.nanmean(segment[:, 1])),
        f"{prefix}_z_mean": float(np.nanmean(segment[:, 2])),
        f"{prefix}_x_std": float(np.nanstd(segment[:, 0])),
        f"{prefix}_y_std": float(np.nanstd(segment[:, 1])),
        f"{prefix}_z_std": float(np.nanstd(segment[:, 2])),
    }


for label_file in label_files:
    participant = label_file.name.split("_")[0]
    participant_dir = label_file.parent

    labels = pd.read_csv(
        label_file,
        header=None,
        names=["start_time", "end_time", "reps", "exercise"]
    )

    sensor_paths = {
        "sw_l_acc": participant_dir / f"{participant}_sw_l_acc.npy",
        "sw_r_acc": participant_dir / f"{participant}_sw_r_acc.npy",
        "sw_l_gyr": participant_dir / f"{participant}_sw_l_gyr.npy",
        "sw_r_gyr": participant_dir / f"{participant}_sw_r_gyr.npy",
    }

    for _, label in labels.iterrows():
        start_time = label["start_time"]
        end_time = label["end_time"]

        row = {
            "participant": participant,
            "start_time": start_time,
            "end_time": end_time,
            "duration": end_time - start_time,
            "reps": label["reps"],
            "exercise": label["exercise"],
        }

        for prefix, path in sensor_paths.items():
            row.update(extract_sensor_features(path, start_time, end_time, prefix))

        rows.append(row)


features = pd.DataFrame(rows)

output_path = OUT_DIR / "mmfit_wearable_features_basic.csv"
features.to_csv(output_path, index=False)

print("Wearable feature table saved to:")
print(output_path)

print("\nShape:")
print(features.shape)

print("\nPreview:")
print(features.head())

print("\nMissing values:")
print(features.isna().sum().sort_values(ascending=False).head(30))

valid_cols = [col for col in features.columns if col.endswith("_valid")]

print("\nValid counts:")
for col in valid_cols:
    print(col)
    print(features[col].value_counts())

print("\nExercise counts:")
print(features["exercise"].value_counts())  