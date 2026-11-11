from pathlib import Path
import numpy as np

RAW_DIR = Path(r"C:\Masters\data\raw\mmfit")
participant = "w00"

for sensor in ["sw_l_hr", "sw_r_hr"]:
    path = RAW_DIR / participant / f"{participant}_{sensor}.npy"
    data = np.load(path, allow_pickle=True)

    print("\nSensor:", sensor)
    print("Path:", path)
    print("Shape:", data.shape)
    print("First 10 rows:")
    print(data[:10])