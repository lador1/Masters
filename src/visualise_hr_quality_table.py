from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

FEATURE_PATH = Path(r"C:\Masters\data\features\mmfit\mmfit_fusion_features_with_hr_cleaned.csv")
FIG_DIR = Path(r"C:\Masters\results\mmfit\figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE = FIG_DIR / "hr_quality_visual_table.png"
OUT_SUMMARY = FIG_DIR / "hr_quality_summary_table.png"

data = pd.read_csv(FEATURE_PATH)

# -------------------------------------------------------
# 1. Summary table
# -------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Total segments",
        "Left HR available",
        "Right HR available",
        "Any HR available",
        "Both HR available",
        "Left/right disagreement",
        "No HR"
    ],
    "Count": [
        len(data),
        int(data["hr_left_available"].sum()),
        int(data["hr_right_available"].sum()),
        int(data["hr_any_available"].sum()),
        int(data["hr_both_available"].sum()),
        int(data["hr_disagreement_flag"].sum()),
        int((data["hr_quality_flag"] == "no_hr").sum())
    ]
})

summary["Percent"] = (summary["Count"] / len(data) * 100).round(1)

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axis("off")

table = ax.table(
    cellText=summary.values,
    colLabels=summary.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

ax.set_title("MM-Fit heart-rate availability and quality summary", fontsize=14, pad=15)

plt.tight_layout()
plt.savefig(OUT_SUMMARY, dpi=300, bbox_inches="tight")
plt.close()

print("Saved summary table:")
print(OUT_SUMMARY)


# -------------------------------------------------------
# 2. Visual table of HR quality by segment
# -------------------------------------------------------

table_cols = [
    "participant",
    "exercise",
    "start_time",
    "end_time",
    "sw_l_hr_mean",
    "sw_r_hr_mean",
    "hr_mean_combined",
    "hr_left_right_diff",
    "hr_quality_flag"
]

hr_table = data[table_cols].copy()

# Round numeric columns
for col in [
    "sw_l_hr_mean",
    "sw_r_hr_mean",
    "hr_mean_combined",
    "hr_left_right_diff"
]:
    hr_table[col] = hr_table[col].round(2)

# Show the most important/problem rows first
hr_table = hr_table.sort_values(
    by="hr_left_right_diff",
    ascending=False
).head(35)

fig, ax = plt.subplots(figsize=(15, 10))
ax.axis("off")

table = ax.table(
    cellText=hr_table.values,
    colLabels=hr_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.4)

ax.set_title(
    "MM-Fit heart-rate quality check: largest left/right disagreements",
    fontsize=14,
    pad=15
)

plt.tight_layout()
plt.savefig(OUT_TABLE, dpi=300, bbox_inches="tight")
plt.close()

print("Saved visual HR table:")
print(OUT_TABLE)

print("\nDone.")