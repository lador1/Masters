from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

FEATURE_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_pose_features_v2_clean.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
OUT_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(FEATURE_PATH)

drop_cols = [
    "participant",
    "start_time",
    "end_time",
    "duration",
    "reps",
    "exercise",
    "pose_valid",
]

feature_cols = [col for col in data.columns if col not in drop_cols]

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
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

labels = sorted(y.unique())
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print("Improved pose-only Random Forest baseline")
print("----------------------------------------")
print("Rows used:", len(data))
print("Features used:", len(feature_cols))
print("Train rows:", len(X_train))
print("Test rows:", len(X_test))
print("Accuracy:", accuracy)
print("Macro F1:", macro_f1)

print("\nClassification report:")
print(report)

print("\nConfusion matrix labels:")
print(labels)

print("\nConfusion matrix:")
print(conf_matrix)

results_summary = pd.DataFrame([
    {
        "model": "RandomForest",
        "modality": "pose_only_v2",
        "rows_used": len(data),
        "features_used": len(feature_cols),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
    }
])

results_summary.to_csv(OUT_DIR / "pose_baseline_v2_summary.csv", index=False)

confusion_df = pd.DataFrame(
    conf_matrix,
    index=labels,
    columns=labels
)

confusion_df.to_csv(OUT_DIR / "pose_baseline_v2_confusion_matrix.csv")

feature_importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

feature_importance.to_csv(OUT_DIR / "pose_baseline_v2_feature_importance.csv", index=False)

print("\nTop 20 feature importances:")
print(feature_importance.head(20))

print("\nSaved:")
print(OUT_DIR / "pose_baseline_v2_summary.csv")
print(OUT_DIR / "pose_baseline_v2_confusion_matrix.csv")
print(OUT_DIR / "pose_baseline_v2_feature_importance.csv")