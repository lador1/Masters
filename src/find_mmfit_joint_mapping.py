from pathlib import Path

REPO_DIR = Path(r"C:\Masters\external\mm-fit-code")

if not REPO_DIR.exists():
    print("Repo folder not found:")
    print(REPO_DIR)
    print("\nYou may need to clone or download the MM-Fit GitHub repo first.")
    raise SystemExit

terms = [
    "joint",
    "joints",
    "skeleton",
    "keypoint",
    "keypoints",
    "pose_3d",
    "pose_2d",
    "openpose",
    "coco",
    "body",
]

file_types = [".py", ".ipynb", ".md", ".txt", ".json", ".yml"]

for path in REPO_DIR.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() not in file_types:
        continue

    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        continue

    hits = []

    for i, line in enumerate(lines):
        lower = line.lower()
        if any(term in lower for term in terms):
            hits.append((i + 1, line.strip()))

    if hits:
        print("\n" + "=" * 90)
        print(path.relative_to(REPO_DIR))
        print("=" * 90)

        for line_number, line in hits[:80]:
            print(f"{line_number}: {line}")