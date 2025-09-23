#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_video.py

📌 전체 파이프라인 (단일 비디오 처리):
- 선택적으로 실행 가능 (frame, sapiens, reextract, overlay)
- 실행 여부를 CSV 컬럼 (frames_done, sapiens_done, reextract_done, overlay_done) 에 기록
"""

import sys
from pathlib import Path
import pandas as pd

# ---------------- 경로 설정 ----------------
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove")
CSV_PATH = BASE_DIR / "data" / "metadata.csv"

# 모듈 경로 추가
sys.path.append(str(BASE_DIR / "py"))

from extract_frames import extract_frames
from extract_keypoints import extract_keypoints
from reextract_missing_keypoints import reextract_missing_keypoints
from create_overlay import create_overlay
from mmpose.apis import init_model as init_pose_estimator


def process_video(video_path: Path,
                  run_frames: bool = True,
                  run_sapiens: bool = True,
                  run_reextract: bool = True,
                  run_overlay: bool = True):
    """단일 비디오 처리 파이프라인"""

    # ---------------- Raw data 존재 여부 확인 ----------------
    if not video_path.exists():
        print(f"[ERROR] Raw video 파일 없음 → {video_path}")
        return  # 🚨 실행 중단

    # ---------------- 경로 설정 ----------------
    rel_path = video_path.relative_to(BASE_DIR / "data" / "0_RAW_DATA").with_suffix("")

    frame_dir    = BASE_DIR / "data" / "1_FRAME"    / rel_path.parent / (rel_path.name + "_frames")
    keypoint_dir = BASE_DIR / "data" / "2_KEYPOINTS"/ rel_path.parent / (rel_path.name + "_JSON")
    mp4_path     = BASE_DIR / "data" / "3_MP4"     / rel_path.parent / (rel_path.name + ".mp4")

    rel_video_path     = video_path.relative_to(BASE_DIR / "data")
    rel_frame_path     = frame_dir.relative_to(BASE_DIR / "data")
    rel_keypoints_path = keypoint_dir.relative_to(BASE_DIR / "data")
    rel_mp4_path       = mp4_path.relative_to(BASE_DIR / "data")

    # ---------------- 출력 디렉토리 생성 보장 ----------------
    frame_dir.parent.mkdir(parents=True, exist_ok=True)
    keypoint_dir.parent.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Video    : {video_path}")
    print(f"[INFO] Frames   : {frame_dir}")
    print(f"[INFO] Keypoints: {keypoint_dir}")
    print(f"[INFO] MP4      : {mp4_path}")

    # ---------------- 상태 초기화 ----------------
    n_frames, n_json, final_json_count = 0, 0, 0
    frames_done = frame_dir.exists() and any(frame_dir.glob("*.jpg"))
    sapiens_done = keypoint_dir.exists() and any(keypoint_dir.glob("*.json"))
    overlay_done = mp4_path.exists()
    reextract_done = False

    # ---------------- 1. 프레임 추출 ----------------
    if run_frames:
        n_frames = extract_frames(str(video_path), str(frame_dir))
        print(f"[INFO] 프레임 추출 완료: {n_frames} frames")
        frames_done = True

    # ---------------- 2. Sapiens 실행 ----------------
    if run_sapiens:
        n_json = extract_keypoints(
            str(frame_dir), str(keypoint_dir),
            det_cfg  = str(BASE_DIR / "sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"),
            det_ckpt = str(BASE_DIR / "sapiens/pose/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"),
            pose_cfg = str(BASE_DIR / "sapiens/pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py"),
            pose_ckpt= str(BASE_DIR / "sapiens/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0"
        )
        print(f"[INFO] Sapiens 추출 완료: {n_json} JSON")
        sapiens_done = True

    # ---------------- 3. 누락 프레임 보정 ----------------
    if run_reextract:
        if n_frames == 0 and frame_dir.exists():
            n_frames = len(list(frame_dir.glob("*.jpg")))

        pose_estimator = init_pose_estimator(
            str(BASE_DIR / "sapiens/pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py"),
            str(BASE_DIR / "sapiens/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"),
            device="cuda:0",
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
        )
        final_json_count = reextract_missing_keypoints(
            file_name = video_path.name,
            frame_dir = str(frame_dir),
            json_dir  = str(keypoint_dir),
            n_extracted_frames = n_frames,
            pose_estimator = pose_estimator
        )
        print(f"[INFO] 누락 보정 후 최종 JSON: {final_json_count}")
        reextract_done = True

    # ---------------- 4. Overlay 생성 ----------------
    if run_overlay:
        create_overlay(str(frame_dir), str(keypoint_dir), str(mp4_path), fps=30)
        print(f"[INFO] Overlay mp4 생성 완료 → {mp4_path}")
        overlay_done = True

    # ---------------- 5. metadata.csv 갱신 ----------------
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
        df = pd.read_csv(CSV_PATH)
        if str(rel_video_path) in df["video_path"].values:
            prev = df[df["video_path"] == str(rel_video_path)].iloc[0].to_dict()
        else:
            prev = {}
    else:
        df = pd.DataFrame()
        prev = {}

    row = {
        "video_path":     str(rel_video_path),
        "frame_path":     str(rel_frame_path),
        "keypoints_path": str(rel_keypoints_path),
        "mp4_path":       str(rel_mp4_path),
        "n_frames":       n_frames if n_frames > 0 else prev.get("n_frames", 0),
        "n_json":         (final_json_count if run_reextract else (n_json if run_sapiens else prev.get("n_json", 0))),
        "frames_done":    frames_done,
        "sapiens_done":   sapiens_done,
        "reextract_done": reextract_done or prev.get("reextract_done", False),
        "overlay_done":   overlay_done
    }

    if not df.empty and str(rel_video_path) in df["video_path"].values:
        for k, v in row.items():
            df.loc[df["video_path"] == row["video_path"], k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] metadata.csv 갱신 완료 → {CSV_PATH}")
