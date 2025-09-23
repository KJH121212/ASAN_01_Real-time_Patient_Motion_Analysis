#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, cv2, json, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# mmdetection, mmpose
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances
import mmpretrain  # VisionTransformer 등록

# ---------------- 경로 ----------------
RAW_DATA_ROOT = Path("../data/new_data/raw_data")   
CSV_PATH = Path("../data/new_data/video_metadata.csv")
OUTPUT_ROOTS = {
    "frames_dir": Path("../data/new_data/frames_output"),
    "keypoints_dir": Path("../data/new_data/keypoints_json"),
}
VIDEO_EXTS = [".mp4", ".MP4", ".mov", ".MOV"]

TARGET_SHORT = 720   # 프레임 리사이즈: 짧은 축을 720으로
JPEG_QUALITY = 80    # JPEG 품질 (속도/용량 최적화)
MAX_WORKERS = 4      # 병렬 프로세스 개수

# Sapiens 모델 설정
DET_CONFIG  = "../sapiens/pose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person_no_nms.py"
DET_CKPT    = "../sapiens/pose/checkpoints/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
POSE_CONFIG = "../sapiens/pose/configs/sapiens_pose/coco/sapiens_0.3b-210e_coco-1024x768.py"
POSE_CKPT   = "../sapiens/pose/checkpoints/sapiens_0.3b/sapiens_0.3b_coco_best_coco_AP_796.pth"

# ---------------- 유틸 ----------------
def to_py(obj):
    """넘파이 객체 → 파이썬 기본 타입 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

def get_video_metadata(video_path: Path):
    """ffprobe로 width, height, fps, frame 수 추출"""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,codec_name,r_frame_rate,duration,nb_frames",
            "-of", "json", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        stream = info["streams"][0]
        num, den = map(int, stream["r_frame_rate"].split('/'))
        fps = num / den if den != 0 else None
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "codec": stream.get("codec_name", None),
            "fps": float(fps) if fps else None,
            "duration_sec": float(stream.get("duration", 0)),
            "num_frames": int(stream.get("nb_frames", 0)) if stream.get("nb_frames", "0").isdigit() else None
        }
    except Exception:
        return {k: None for k in ["width","height","codec","fps","duration_sec","num_frames"]}

# ---------------- Frame 추출 ----------------
def extract_frames(video_path, frame_dir, target_short=720, jpeg_quality=80):
    """비디오에서 프레임 추출 (짧은 축을 720으로 맞춰 리사이즈, JPEG 품질 조정)"""
    video_path, frame_dir = Path(video_path), Path(frame_dir)
    if frame_dir.exists():
        for f in frame_dir.glob("*.jpg"):
            f.unlink()  # 기존 jpg 삭제
    frame_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return f"[SKIP] 열기 실패: {video_path}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 짧은 축을 720으로 맞춤
    if w <= h:
        scale = target_short / w
    else:
        scale = target_short / h
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret: break
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out_path = frame_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(out_path), resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

    cap.release()
    return f"[DONE] {video_path.name} {w}x{h} → {new_w}x{new_h}, 총 {n_frames} 프레임"

# ---------------- CSV 업데이트 ----------------
def update_metadata_csv():
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
    else:
        df = pd.DataFrame(columns=[
            "file_name","video_path","subdir",
            "width","height","num_frames","fps","duration_sec","codec",
            "frames_dir","keypoints_dir",
            "frames_verified","sapiens_done","exists"
        ])

    df["exists"] = False
    video_files = [f for f in RAW_DATA_ROOT.rglob("*") if f.suffix in VIDEO_EXTS]

    for video_path in video_files:
        file_name = video_path.name
        rel_video_path = os.path.relpath(video_path, start=Path.cwd())
        subdir = str(video_path.relative_to(RAW_DATA_ROOT).parent)

        # ✅ video_path 기준으로 확인
        if rel_video_path in df["video_path"].values:
            idx = df.index[df["video_path"] == rel_video_path][0]
            meta = get_video_metadata(video_path)
            df.loc[idx, ["file_name","subdir"]] = [file_name, subdir]
            df.loc[idx, "exists"] = True
            for k, v in meta.items(): df.loc[idx, k] = v
        else:
            meta = get_video_metadata(video_path)
            new_row = {
                "file_name": file_name,
                "video_path": rel_video_path,
                "subdir": subdir,
                **meta,
                "frames_dir": str(OUTPUT_ROOTS["frames_dir"]/subdir/f"{Path(file_name).stem}_frames"),
                "keypoints_dir": str(OUTPUT_ROOTS["keypoints_dir"]/subdir),
                "frames_verified": False,
                "sapiens_done": False,
                "exists": True
            }
            df.loc[len(df)] = new_row

    df.to_csv(CSV_PATH, index=False)
    return df

# ---------------- Sapiens 실행 ----------------
def run_sapiens_on_frames(video_row, detector, pose_estimator, df, idx):
    video_filename = video_row["file_name"]
    if video_row["sapiens_done"]: return
    frame_dir = Path(video_row["frames_dir"])
    if not frame_dir.exists(): return
    json_dir = Path(video_row["keypoints_dir"])/f"{Path(video_filename).stem}_JSON"
    if json_dir.exists():
        for f in json_dir.glob("*.json"):
            f.unlink()
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

    df.loc[idx,"sapiens_done"] = True

# ---------------- 메인 ----------------
def main():
    df = update_metadata_csv()

    # ✅ 병렬 프레임 추출
    tasks = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx,row in df.iterrows():
            if row["exists"] and not row["frames_verified"]:
                tasks[executor.submit(
                    extract_frames,
                    row["video_path"],
                    row["frames_dir"],
                    TARGET_SHORT,
                    JPEG_QUALITY
                )] = idx

        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Frame Extraction"):
            idx = tasks[future]
            print(future.result())
            df.loc[idx,"frames_verified"] = True

    # ✅ Sapiens 실행
    detector = init_detector(DET_CONFIG,DET_CKPT,device="cuda:0")
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        POSE_CONFIG,POSE_CKPT,device="cuda:0",
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )
    for idx,row in df.iterrows():
        if row["exists"] and df.loc[idx,"frames_verified"] and not row["sapiens_done"]:
            run_sapiens_on_frames(row, detector, pose_estimator, df, idx)

    df.to_csv(CSV_PATH,index=False)
    print(f"[INFO] CSV 업데이트 완료: {CSV_PATH}")

if __name__=="__main__":
    main()
