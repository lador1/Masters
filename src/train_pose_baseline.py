from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

FEATURE_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_pose_features_clean.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(FEATURE_PATH)

feature_cols = [
    "pose_movement_mean",
    "pose_movement_std",
    "pose_range_mean",
    "pose_range_max",
    "pose_stability_proxy",
]

X = data[feature_cols]
y = data["exercise"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))

print("Pose-only Random Forest baseline")
print("--------------------------------")
print("Rows used:", len(data))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)

print("\nClassification report:")
print(report)

print("\nConfusion matrix labels:")
print(sorted(y.unique()))

print("\nConfusion matrix:")
print(conf_matrix)

# Save outputs
results_summary = pd.DataFrame([
    {
        "model": "RandomForest",
        "modality": "pose_only",
        "rows_used": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
])

results_summary.to_csv(OUT_DIR / "pose_baseline_summary.csv", index=False)

confusion_df = pd.DataFrame(
    conf_matrix,
    index=sorted(y.unique()),
    columns=sorted(y.unique())
)

confusion_df.to_csv(OUT_DIR / "pose_baseline_confusion_matrix.csv")

print("\nSaved:")
print(OUT_DIR / "pose_baseline_summary.csv")
print(OUT_DIR / "pose_baseline_confusion_matrix.csv")