import pandas as pd
from pathlib import Path

BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove")
RAW_DIR = BASE_DIR / "data" / "0_RAW_DATA"
CSV_PATH = BASE_DIR / "data" / "metadata.csv"

def update_metadata():
    # CSV 로드 (없으면 새로 생성)
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=[
            "video_path", "frame_path", "keypoints_path", "mp4_path",
            "n_frames", "n_json", "frames_done", "sapiens_done",
            "reextract_done", "overlay_done"
        ])

    # RAW_DATA 내부 모든 영상 탐색
    video_files = list(RAW_DIR.rglob("*.MP4")) + list(RAW_DIR.rglob("*.MOV"))

    new_rows = []
    for video_path in video_files:
        rel_video_path = video_path.relative_to(BASE_DIR / "data")

        # 이미 metadata.csv에 존재하는지 확인
        if str(rel_video_path) in df["video_path"].values:
            continue  # 이미 있으면 skip

        # 기본 경로 설정
        rel_path = rel_video_path.with_suffix("")
        frame_path = "1_FRAME/" + str(rel_path.parent / (rel_path.name + "_frames"))
        keypoints_path = "2_KEYPOINTS/" + str(rel_path.parent / (rel_path.name + "_JSON"))
        mp4_path = "3_MP4/" + str(rel_path.parent / (rel_path.name + ".mp4"))

        new_rows.append({
            "video_path": str(rel_video_path),
            "frame_path": frame_path,
            "keypoints_path": keypoints_path,
            "mp4_path": mp4_path,
            "n_frames": 0,
            "n_json": 0,
            "frames_done": False,
            "sapiens_done": False,
            "reextract_done": False,
            "overlay_done": False
        })

    # 새로운 row 추가
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
        print(f"[INFO] metadata.csv 업데이트 완료: {len(new_rows)}개 추가됨")
    else:
        print("[INFO] 새로운 파일 없음")

    return df

# 실행 예시
if __name__ == "__main__":
    df_updated = update_metadata()
    print(df_updated.tail())
