# This script processes the raw MM-Fit label files to create a consolidated summary CSV file. It reads all label files, extracts relevant information, and compiles it into a single DataFrame for analysis and future use in model training or evaluation. The summary includes participant IDs, exercise types, durations, and repetition counts.


from pathlib import Path
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\processed\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_labels = []

label_files = sorted(RAW_DIR.rglob("*_labels.csv"))

for label_file in label_files:
    participant = label_file.name.split("_")[0]

    labels = pd.read_csv(
        label_file,
        header=None,
        names=["start_time", "end_time", "reps", "exercise"]
    )

    labels["participant"] = participant
    labels["duration"] = labels["end_time"] - labels["start_time"]

    all_labels.append(labels)

label_data = pd.concat(all_labels, ignore_index=True)

label_data = label_data[
    ["participant", "start_time", "end_time", "duration", "reps", "exercise"]
]

output_path = OUT_DIR / "mmfit_label_summary.csv"
label_data.to_csv(output_path, index=False)

print("Label summary saved to:")
print(output_path)

print("\nPreview:")
print(label_data.head(20))

print("\nTotal labelled segments:")
print(len(label_data))

print("\nSegments per participant:")
print(label_data["participant"].value_counts().sort_index())

print("\nExercise counts:")
print(label_data["exercise"].value_counts())

print("\nDuration summary:")
print(label_data["duration"].describe())