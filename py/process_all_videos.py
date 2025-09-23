import sys
from pathlib import Path
import pandas as pd

# ---------------- 기본 경로 ----------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove")
DATA_DIR = BASE_DIR / "data"
CSV_PATH = DATA_DIR / "metadata.csv"

# ---------------- py 폴더 import ----------------
sys.path.append(str(BASE_DIR / "py"))
from process_video import process_video

# ---------------- metadata 불러오기 ----------------
df = pd.read_csv(CSV_PATH)

print(f"[INFO] metadata.csv 내 총 비디오 개수: {len(df)}")

# ---------------- 전체 비디오 처리 ----------------
for _, row in df.iterrows():
    video_path = DATA_DIR / row["video_path"]

    if not video_path.exists():
        print(f"[WARN] Raw video 없음 → {video_path}")
        continue

    run_frames     = not bool(row.get("frames_done", False))
    run_sapiens    = not bool(row.get("sapiens_done", False))
    run_reextract  = not bool(row.get("reextract_done", False))
    run_overlay    = not bool(row.get("overlay_done", False))

    print(f"\n[PROCESS] {video_path}")
    print(f"  frames={run_frames}, sapiens={run_sapiens}, reextract={run_reextract}, overlay={run_overlay}")

    process_video(
        video_path,
        run_frames=run_frames,
        run_sapiens=run_sapiens,
        run_reextract=run_reextract,
        run_overlay=run_overlay
    )
