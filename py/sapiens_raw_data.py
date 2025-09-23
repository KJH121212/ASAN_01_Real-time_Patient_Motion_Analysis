#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sapiens_raw_data.py

📌 전체 파이프라인:
1. RAW_DATA_ROOT 내부에서 mp4/mov 영상 파일 탐색
2. CSV(video_metadata.csv)와 비교 → 없는 파일은 신규 추가
   - video_path 실제 파일 없으면 exists=0
   - RAW_DATA_ROOT에 새 파일 있으면 CSV에 추가 및 즉시 저장
3. frames_verified == 0 인 경우:
   - 기존 프레임 폴더 전체 삭제
   - 영상으로부터 720p 다운샘플링하여 프레임 재추출
   - 추출 완료 후 frames_verified=1, n_extracted_frames 업데이트
   - 프레임 추출이 끝나면 CSV 저장
4. frames_verified == 1 인 경우: 프레임 추출 스킵
5. Sapiens 모델 실행하여 keypoints_json 저장
   - 실행 완료 시 해당 row를 즉시 CSV에 저장
"""

import os, cv2, json, subprocess, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------- 외부 라이브러리 ----------------
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
import mmpretrain  # VisionTransformer 등록

# ---------------- 경로 설정 ----------------
RAW_DATA_ROOT = Path("../data/new_data/raw_data")
CSV_PATH = Path("../data/new_data/video_metadata.csv")
OUTPUT_ROOTS = {
    "frames_dir": Path("../data/new_data/frames_output"),
    "keypoints_dir": Path("../data/new_data/keypoints_json"),
}
VIDEO_EXTS = [".mp4", ".MP4", ".mov", ".MOV"]

# ---------------- 프레임 추출 옵션 ----------------
TARGET_SHORT = 720
JPEG_QUALITY = 80
MAX_WORKERS = 4

# ---------------- Sapiens 모델 설정 ----------------
DET_CONFIG  = "../sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"
DET_CKPT    = "../sapiens/pose/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG = "../sapiens/pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py"
POSE_CKPT   = "../sapiens/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"

# ---------------- 유틸 함수 ----------------
def to_py(obj):
    """넘파이 객체를 JSON 직렬화 가능한 파이썬 타입으로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

# ---------------- Frame 추출 ----------------
def extract_frames(video_path, frame_dir, target_short=720, jpeg_quality=80):
    """프레임 재추출 (폴더 삭제 후 720p 리사이즈 저장, 추출된 프레임 수 반환)"""
    video_path, frame_dir = Path(video_path), Path(frame_dir)
    if frame_dir.exists():
        shutil.rmtree(frame_dir)  # ✅ 기존 폴더 삭제
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[SKIP] 열기 실패: {video_path}", 0

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scale = target_short / w if w <= h else target_short / h
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    extracted_count = 0
    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out_path = frame_dir / f"{idx:06d}.jpg"
        if cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]):
            extracted_count += 1

    cap.release()
    return f"[DONE] {video_path.name} {w}x{h} → {new_w}x{new_h}, 총 {extracted_count} 프레임", extracted_count

# ---------------- CSV 업데이트 ----------------
def update_metadata_csv():
    """CSV 갱신 (존재 여부만 체크, 메타데이터 갱신은 안 함)"""
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    else:
        df = pd.DataFrame(columns=[
            "file_name","video_path","subdir",
            "width","height","num_frames","fps","duration_sec","codec",
            "frames_dir","n_extracted_frames","keypoints_dir",
            "frames_verified","sapiens_done","exists"
        ])

    # 존재 여부만 체크
    for idx, row in df.iterrows():
        vpath = Path(row["video_path"])
        df.loc[idx, "exists"] = int(vpath.exists())

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV 존재 여부 갱신 완료: {CSV_PATH}")
    return df

# ---------------- Sapiens 실행 ----------------
def run_sapiens_on_frames(video_row, detector, pose_estimator, df, idx):
    """Sapiens 모델 실행 (완료 후 즉시 CSV 저장)"""
    video_filename = video_row["file_name"]
    if video_row["sapiens_done"] == 1: return
    frame_dir = Path(video_row["frames_dir"])
    if not frame_dir.exists(): return

    json_dir = Path(video_row["keypoints_dir"])/f"{Path(video_filename).stem}_JSON"
    json_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frame_dir.glob("*.jpg"))

    for idx_frame, fpath in enumerate(tqdm(frames, desc=video_filename, unit="frame")):
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            det = inference_detector(detector, img_rgb)
            pred = det.pred_instances.cpu().numpy()
            keep = (pred.labels==0) & (pred.scores>0.5)
            bbs = np.concatenate((pred.bboxes, pred.scores[:,None]), axis=1)[keep]
            if len(bbs)==0: continue
            bbs = bbs[nms(bbs,0.5),:4]

            pose_results = inference_topdown(pose_estimator,img_rgb,bbs)
            data_sample = merge_data_samples(pose_results)
            inst = data_sample.get("pred_instances",None)
            if inst is None: continue
            inst_list = split_instances(inst)

            payload = dict(frame_index=idx_frame,video_name=video_filename,
                           meta_info=pose_estimator.dataset_meta,instance_info=inst_list)
            json_path = json_dir/f"{idx_frame:06d}.json"
            with open(json_path,"w",encoding="utf-8") as f:
                json.dump(to_py(payload),f,ensure_ascii=False,indent=2)
        except Exception as e:
            print(f"[ERROR] {video_filename} frame {idx_frame} → {e}")

    # ✅ 완료 후 CSV 업데이트
    df.loc[idx,"sapiens_done"] = 1
    df.to_csv(CSV_PATH,index=False,encoding="utf-8-sig")
    print(f"[INFO] CSV 업데이트 (sapiens_done=1): {video_filename}")

# ---------------- 메인 ----------------
def main():
    # 1) CSV 업데이트 (존재 여부만)
    df = update_metadata_csv()

    # 2) 프레임 추출
    tasks = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, row in df.iterrows():
            if row["exists"] == 1 and row["frames_verified"] == 0:
                tasks[executor.submit(
                    extract_frames, row["video_path"], row["frames_dir"], TARGET_SHORT, JPEG_QUALITY
                )] = idx

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Frame Extraction"):
            idx = tasks[future]
            msg, n_extracted = future.result()
            print(msg)
            df.loc[idx,"frames_verified"] = 1
            df.loc[idx,"n_extracted_frames"] = n_extracted   # ✅ 추출된 프레임 수 업데이트
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")

    print(f"[INFO] CSV 업데이트 완료 (프레임 추출 후): {CSV_PATH}")

    # Sapiens 실행
    detector = init_detector(DET_CONFIG,DET_CKPT,device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        POSE_CONFIG,POSE_CKPT,device="cuda:0",
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )

    # 1) sample_data 우선 실행
    for idx,row in df.iterrows():
        if "sample_data" in str(row["video_path"]) and row["exists"] == 1 and row["frames_verified"] == 1 and row["sapiens_done"] == 0:
            run_sapiens_on_frames(row, detector, pose_estimator, df, idx)

    # 2) 나머지 실행
    for idx,row in df.iterrows():
        if "sample_data" not in str(row["video_path"]) and row["exists"] == 1 and row["frames_verified"] == 1 and row["sapiens_done"] == 0:
            run_sapiens_on_frames(row, detector, pose_estimator, df, idx)


# ---------------- 실행 ----------------
if __name__=="__main__":
    main()
