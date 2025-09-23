#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, cv2, json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
import mmpretrain  # VisionTransformer 등록

# ---------------- 모델 설정 ----------------
DET_CONFIG  = "../sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"
DET_CKPT    = "../sapiens/pose/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG = "../sapiens/pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py"
POSE_CKPT   = "../sapiens/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"
CSV_PATH    = "../data/new_data/new_video_metadata.csv"
OUTPUT_ROOT = "../data/new_data/keypoints_json"
SKIP_FILE   = "./new_skip.txt"   # ✅ 완료된 비디오 기록 파일

# ---------------- 유틸 ----------------
def to_py(obj):
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def load_skip_list():
    """이미 완료된 비디오 목록 불러오기"""
    if not Path(SKIP_FILE).exists():
        return set()
    with open(SKIP_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def save_skip(video_name):
    """완료된 비디오를 new_skip.txt에 추가"""
    with open(SKIP_FILE, "a", encoding="utf-8") as f:
        f.write(video_name + "\n")

# ---------------- 처리 함수 ----------------
def run_sapiens_on_frames(video_row, detector, pose_estimator, skip_list):
    """단일 비디오(frame_dir 사용) → JSON 저장"""
    video_filename = video_row["filename"]
    if video_filename in skip_list:
        print(f"[SKIP] 이미 완료됨 (skip list): {video_filename}")
        return

    frame_dir = Path(video_row["frame_dir"])
    if pd.isna(video_row["frame_dir"]) or not frame_dir.exists():
        print(f"[SKIP] 프레임 없음: {video_filename}")
        return

    # 출력 경로
    rel_path = Path(video_row["folder"])
    json_dir = Path(OUTPUT_ROOT) / rel_path / f"{Path(video_filename).stem}_JSON"
    json_dir.mkdir(parents=True, exist_ok=True)

    # 프레임 읽기
    frames = sorted(frame_dir.glob("*.jpg"))
    if not frames:
        print(f"[SKIP] 프레임 비어있음: {frame_dir}")
        return

    print(f"[INFO] {video_filename} | 프레임 {len(frames)}개 처리 예정")

    ok, fail = 0, 0
    for idx, fpath in enumerate(tqdm(frames, desc=video_filename, unit="frame")):
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            det = inference_detector(detector, img_rgb)
            pred = det.pred_instances.cpu().numpy()
            if len(pred.bboxes) == 0:
                continue
            bbs = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)
            keep = (pred.labels == 0) & (pred.scores > 0.5)
            bbs = bbs[keep]
            if len(bbs) == 0:
                continue
            bbs = bbs[nms(bbs, 0.5), :4]

            pose_results = inference_topdown(pose_estimator, img_rgb, bbs)
            data_sample = merge_data_samples(pose_results)
            inst = data_sample.get("pred_instances", None)
            if inst is None:
                continue

            inst_list = split_instances(inst)
            payload = dict(
                frame_index=idx,
                video_name=video_filename,
                meta_info=pose_estimator.dataset_meta,
                instance_info=inst_list
            )
            json_path = json_dir / f"{idx:06d}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(to_py(payload), f, ensure_ascii=False, indent=2)

            ok += 1
        except Exception as e:
            fail += 1
            print(f"[ERROR] {video_filename} frame {idx} → {e}")

    print(f"[DONE] {video_filename} 성공 {ok}, 실패 {fail}, JSON 저장: {json_dir}")
    # ✅ 완료된 경우 skip 리스트에 추가
    save_skip(video_filename)

# ---------------- 메인 ----------------
def main():
    # CSV 로드
    df = pd.read_csv(CSV_PATH)

    # sample_data와 나머지 분리
    df_sample = df[df["folder"].astype(str).str.startswith("sample_data/")]
    df_rest   = df[~df["folder"].astype(str).str.startswith("sample_data/")]
    print(f"[INFO] sample_data 내 영상 {len(df_sample)}개, 나머지 {len(df_rest)}개")

    # skip list 로드
    skip_list = load_skip_list()
    print(f"[INFO] skip 목록 {len(skip_list)}개 불러옴")

    # 모델 초기화 (한 번만)
    detector = init_detector(DET_CONFIG, DET_CKPT, device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        POSE_CONFIG, POSE_CKPT, device="cuda:0",
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )

    # 1) sample_data 먼저 실행
    for _, row in df_sample.iterrows():
        run_sapiens_on_frames(row, detector, pose_estimator, skip_list)

    # 2) 나머지 실행
    for _, row in df_rest.iterrows():
        run_sapiens_on_frames(row, detector, pose_estimator, skip_list)


if __name__ == "__main__":
    main()
