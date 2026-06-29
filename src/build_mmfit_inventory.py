# This script says what files exist in the raw MM-Fit data folder and creates an inventory CSV file with details about each file. This helps us understand the dataset structure and plan our processing steps.
from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
OUT_DIR = Path(r"C:\Masters\data\processed\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for file in sorted(RAW_DIR.rglob("*")):
    if file.is_dir():
        continue

    participant = file.name.split("_")[0]

    row = {
        "participant": participant,
        "filename": file.name,
        "path": str(file),
        "extension": file.suffix,
        "modality": None,
        "side": None,
        "signal": None,
        "shape": None,
        "dtype": None,
    }

    name = file.stem

    if file.name.endswith("_labels.csv"):
        row["modality"] = "labels"
        row["signal"] = "exercise_segments"

        labels = pd.read_csv(
            file,
            header=None,
            names=["start_time", "end_time", "reps", "exercise"]
        )

        row["shape"] = str(labels.shape)
        row["dtype"] = "csv"

    elif file.suffix == ".npy":
        arr = np.load(file, allow_pickle=True)
        row["shape"] = str(arr.shape)
        row["dtype"] = str(arr.dtype)

        parts = name.split("_")

        if "pose" in parts:
            row["modality"] = "pose"
            if "2d" in parts:
                row["signal"] = "pose_2d"
            elif "3d" in parts:
                row["signal"] = "pose_3d"

        elif len(parts) >= 4:
            row["modality"] = parts[1]
            row["side"] = parts[2]
            row["signal"] = parts[3]

    rows.append(row)

inventory = pd.DataFrame(rows)

output_path = OUT_DIR / "mmfit_file_inventory.csv"
inventory.to_csv(output_path, index=False)

print("Inventory saved to:")
print(output_path)

print("\nInventory preview:")
print(inventory.head(20))

print("\nFiles by modality:")
print(inventory["modality"].value_counts(dropna=False))

print("\nParticipants found:")
print(sorted(inventory["participant"].unique()))