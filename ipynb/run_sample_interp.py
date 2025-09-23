#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm

# ==============================
# ✅ 보간 + Overlay 함수 정의
# ==============================
def process_video_with_interpolation(frame_dir, json_dir, output_mp4, overlay_dir, interp_json_dir, csv_path):
    frame_dir = Path(frame_dir)
    json_dir = Path(json_dir)
    overlay_dir = Path(overlay_dir)
    interp_json_dir = Path(interp_json_dir)

    overlay_dir.mkdir(parents=True, exist_ok=True)
    interp_json_dir.mkdir(parents=True, exist_ok=True)

    # ----- 파일 로드 -----
    frames = sorted(frame_dir.glob("*.jpg"))
    jsons = sorted(json_dir.glob("*.json"))

    def get_index(fname: Path):
        return int(fname.stem)

    frame_ids = set(get_index(f) for f in frames)
    json_ids = set(get_index(j) for j in jsons)

    missing_json = sorted(frame_ids - json_ids)
    if len(jsons) == 0:
        print(f"[SKIP] JSON 없음 → {frame_dir}")
        return

    # meta_info는 첫 JSON에서 가져옴
    with open(jsons[0], "r", encoding="utf-8") as f:
        meta_info = json.load(f)["meta_info"]

    # JSON 로드 dict
    json_data = {get_index(j): json.load(open(j, "r", encoding="utf-8")) for j in jsons}

    # ----- 보간 -----
    interp_data = {}
    for idx in missing_json:
        prev_idx = max([j for j in json_ids if j < idx], default=None)
        next_idx = min([j for j in json_ids if j > idx], default=None)
        if prev_idx is None or next_idx is None:
            continue

        prev_kps = np.array(json_data[prev_idx]["instance_info"][0]["keypoints"])
        next_kps = np.array(json_data[next_idx]["instance_info"][0]["keypoints"])

        alpha = (idx - prev_idx) / (next_idx - prev_idx)
        interp_kps = (1 - alpha) * prev_kps + alpha * next_kps

        interp_json = {
            "frame_index": idx,
            "video_name": json_data[prev_idx]["video_name"],
            "meta_info": meta_info,
            "instance_info": [{
                "keypoints": interp_kps.tolist()
            }]
        }
        interp_data[idx] = interp_json

    print(f"총 프레임 수: {len(frame_ids)} | JSON 수: {len(json_ids)} | 보간 필요: {len(interp_data)}")

    # ----- mp4 저장 -----
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, _ = cv2.imread(str(frames[0])).shape
    out = cv2.VideoWriter(str(output_mp4), fourcc, 30, (w, h))

    for fpath in tqdm(frames, desc=f"Overlay MP4 {frame_dir.stem}"):
        idx = get_index(fpath)
        frame = cv2.imread(str(fpath))

        # JSON 있으면 그대로, 없으면 보간
        if idx in json_data:
            kps = np.array(json_data[idx]["instance_info"][0]["keypoints"])
            color = (0, 255, 0)  # green
        elif idx in interp_data:
            kps = np.array(interp_data[idx]["instance_info"][0]["keypoints"])
            color = (0, 0, 255)  # red
        else:
            out.write(frame)
            continue

        # skeleton 링크
        for p1, p2 in meta_info["skeleton_links"]:
            if kps[p1][0] > 0 and kps[p2][0] > 0:
                cv2.line(frame,
                         (int(kps[p1][0]), int(kps[p1][1])),
                         (int(kps[p2][0]), int(kps[p2][1])),
                         (255, 255, 0), 2)

        # keypoints + 번호
        for kp_id, (x, y) in enumerate(kps):
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
            cv2.putText(frame, str(kp_id), (int(x)+5, int(y)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        out.write(frame)

    out.release()
    print(f"[DONE] Overlay 비디오 저장 완료: {output_mp4}")

    # ----- 보간 프레임만 저장 -----
    for idx, data in tqdm(interp_data.items(), desc=f"Overlay+JSON 보간 프레임 {frame_dir.stem}"):
        fpath = frame_dir / f"{idx:06d}.jpg"
        if not fpath.exists():
            continue

        frame = cv2.imread(str(fpath))
        kps = np.array(data["instance_info"][0]["keypoints"])

        for p1, p2 in meta_info["skeleton_links"]:
            if kps[p1][0] > 0 and kps[p2][0] > 0:
                cv2.line(frame,
                         (int(kps[p1][0]), int(kps[p1][1])),
                         (int(kps[p2][0]), int(kps[p2][1])),
                         (255, 255, 0), 2)

        for kp_id, (x, y) in enumerate(kps):
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
            cv2.putText(frame, str(kp_id), (int(x)+5, int(y)+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        cv2.imwrite(str(overlay_dir / f"{idx:06d}.jpg"), frame)

        # JSON 저장
        with open(interp_json_dir / f"{idx:06d}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 보간 프레임 Overlay 저장: {overlay_dir}")
    print(f"[DONE] 보간 JSON 저장: {interp_json_dir}")

    # ----- CSV 업데이트 -----
    df = pd.read_csv(csv_path)
    video_name = frame_dir.stem.replace("_frames","")
    mask = df["filename"].str.contains(video_name)

    if "interp_json_dir" not in df.columns:
        df["interp_json_dir"] = None
    if "interp_overlay_dir" not in df.columns:
        df["interp_overlay_dir"] = None
    if "interp_mp4" not in df.columns:
        df["interp_mp4"] = None

    if mask.any():
        if pd.isna(df.loc[mask, "interp_json_dir"]).all():
            df.loc[mask, "interp_json_dir"] = str(interp_json_dir)
        if pd.isna(df.loc[mask, "interp_overlay_dir"]).all():
            df.loc[mask, "interp_overlay_dir"] = str(overlay_dir)
        if pd.isna(df.loc[mask, "interp_mp4"]).all():
            df.loc[mask, "interp_mp4"] = str(output_mp4)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[CSV] {csv_path} 업데이트 완료")


# ==============================
# ✅ sample_data 전체 실행
# ==============================
def run_all_sample_data():
    CSV_PATH = "../data/new_data/new_video_metadata.csv"
    df = pd.read_csv(CSV_PATH)

    df_sample = df[df["folder"].astype(str).str.startswith("sample_data/")]
    print(f"[INFO] sample_data 내 총 {len(df_sample)}개 영상 처리 예정")

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Sample Data Processing"):
        video_name = Path(row["filename"]).stem

        frame_dir = row["frame_dir"]
        json_dir = row["json_dir"]

        overlay_mp4 = f"../data/new_data/overlay_video/{row['folder']}/{video_name}_interp_overlay.mp4"
        overlay_dir = f"../data/new_data/overlay_interp_frames/{row['folder']}/{video_name}_interp"
        interp_json_dir = f"../data/new_data/interp_json/{row['folder']}/{video_name}_interpJSON"

        if pd.isna(frame_dir) or pd.isna(json_dir):
            print(f"[SKIP] frame/json 경로 없음 → {row['filename']}")
            continue

        process_video_with_interpolation(
            frame_dir=frame_dir,
            json_dir=json_dir,
            output_mp4=overlay_mp4,
            overlay_dir=overlay_dir,
            interp_json_dir=interp_json_dir,
            csv_path=CSV_PATH
        )

if __name__ == "__main__":
    run_all_sample_data()
