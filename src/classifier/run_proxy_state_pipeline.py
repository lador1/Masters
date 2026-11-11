import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Masters\src\classifier")
SCRIPTS = [
    ROOT / "audit_squat_proxy_dataset.py",
    ROOT / "build_squat_proxy_modelling_dataset.py",
    ROOT / "train_squat_proxy_pose_only.py",
    ROOT / "train_squat_proxy_wearable_hr.py",
    ROOT / "train_squat_proxy_fusion.py",
    ROOT / "compare_squat_proxy_models.py",
    ROOT / "analyse_proxy_fusion_feature_importance.py",
    ROOT / "run_proxy_ablation_analysis.py",
]

for script in SCRIPTS:
    if not script.exists():
        raise FileNotFoundError(f"Missing pipeline script: {script}")
    print(f"Running: {script.name}")
    result = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"Pipeline failed while running {script.name} (exit code {result.returncode})")

print("PROXY STATE PIPELINE COMPLETE")
print("Main result files:")
for path in [
    Path(r"C:\Masters\results\mmfit\proxy_state_audit_summary.csv"),
    Path(r"C:\Masters\results\mmfit\squat_proxy_modelling_dataset.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_pose_only_summary.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_wearable_hr_summary.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_fusion_summary.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_model_comparison.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_fusion_feature_importance_aggregated.csv"),
    Path(r"C:\Masters\results\mmfit\proxy_ablation_results.csv"),
]:
    print(path)
