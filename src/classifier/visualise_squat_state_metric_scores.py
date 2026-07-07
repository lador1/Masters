from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

INPUT_PATH = Path(r"C:\Masters\results\mmfit\squat_state_metric_scores.csv")
OUT_DIR = Path(r"C:\Masters\results\mmfit")
FIGURE_DIR = OUT_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(INPUT_PATH)

# -----------------------------
# Plot 1: state counts
# -----------------------------
state_counts = data["derived_state"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(state_counts.index, state_counts.values)
plt.title("Derived squat movement-state counts")
plt.xlabel("Derived state")
plt.ylabel("Number of participants")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

state_count_path = FIGURE_DIR / "squat_derived_state_counts.png"
plt.savefig(state_count_path, dpi=300)
plt.close()

# -----------------------------
# Plot 2: participant score table
# -----------------------------
plot_data = data[
    [
        "participant",
        "movement_strength_score",
        "movement_variability_score",
        "engagement_like_score",
        "fatigue_like_score",
        "frustration_like_score",
        "derived_state",
    ]
].copy()

plot_data = plot_data.round(3)

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

table = ax.table(
    cellText=plot_data.values,
    colLabels=plot_data.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.title("Squat derived movement-state scores by participant", fontsize=14, pad=20)
plt.tight_layout()

score_table_path = FIGURE_DIR / "squat_state_scores_table.png"
plt.savefig(score_table_path, dpi=300)
plt.close()

# -----------------------------
# Plot 3: movement strength by participant
# -----------------------------
sorted_data = data.sort_values("movement_strength_score")

plt.figure(figsize=(12, 6))
plt.bar(sorted_data["participant"], sorted_data["movement_strength_score"])
plt.axhline(0, linestyle="--")
plt.title("Squat movement strength score by participant")
plt.xlabel("Participant")
plt.ylabel("Movement strength score")
plt.tight_layout()

strength_path = FIGURE_DIR / "squat_movement_strength_by_participant.png"
plt.savefig(strength_path, dpi=300)
plt.close()

# -----------------------------
# Plot 4: movement variability by participant
# -----------------------------
sorted_data = data.sort_values("movement_variability_score")

plt.figure(figsize=(12, 6))
plt.bar(sorted_data["participant"], sorted_data["movement_variability_score"])
plt.axhline(0, linestyle="--")
plt.title("Squat movement variability score by participant")
plt.xlabel("Participant")
plt.ylabel("Movement variability score")
plt.tight_layout()

variability_path = FIGURE_DIR / "squat_movement_variability_by_participant.png"
plt.savefig(variability_path, dpi=300)
plt.close()

print("Saved figures:")
print(state_count_path)
print(score_table_path)
print(strength_path)
print(variability_path)

print("\nDerived state counts:")
print(state_counts)

print("\nParticipant states:")
print(data[["participant", "derived_state"]])