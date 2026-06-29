# This script is for inspecting the raw MM-Fit data files to understand their structure and contents.

from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")

print("MM-Fit raw folder exists:", RAW_DIR.exists())
print("MM-Fit raw folder:", RAW_DIR)

label_files = sorted(RAW_DIR.rglob("*_labels.csv"))

print("\nLabel files found:", len(label_files))

label_file = label_files[0]
participant = label_file.name.split("_")[0]

print("\nUsing participant:", participant)
print("Using label file:", label_file)

labels = pd.read_csv(
    label_file,
    header=None,
    names=["start_time", "end_time", "reps", "exercise"]
)

print("\nLabels preview:")
print(labels.head())

print("\nLabel columns:")
print(labels.columns.tolist())

print("\nExercise counts:")
print(labels["exercise"].value_counts())

npy_files = sorted(RAW_DIR.rglob(f"{participant}_*.npy"))

print(f"\nNPY file shapes for {participant}:")
for file in npy_files:
    arr = np.load(file, allow_pickle=True)
    print(file.name, arr.shape, arr.dtype)