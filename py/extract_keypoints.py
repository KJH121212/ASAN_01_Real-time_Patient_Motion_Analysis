#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sapiens 모델을 이용한 Keypoints 추출 (Batch 지원)
- 입력: 프레임 디렉토리
- 출력: 프레임별 JSON (keypoints, skeleton 포함)
"""

import cv2, json, shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_estimator, inference_topdown
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples, split_instances

# ----------------------------
# numpy → json 직렬화 변환
# ----------------------------
def to_py(obj):
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj

# ----------------------------
# Keypoints 추출 (Batch 버전)
# ----------------------------
def extract_keypoints(frame_dir: str, json_dir: str,
                      det_cfg: str, det_ckpt: str,
                      pose_cfg: str, pose_ckpt: str,
                      device: str = "cuda:0",
                      batch_size: int = 8) -> int:
    frame_dir, json_dir = Path(frame_dir), Path(json_dir)

    # 🔹 JSON 폴더 초기화
    if json_dir.exists():
        shutil.rmtree(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    # 🔹 Detector & Pose Estimator 초기화
    detector = init_detector(det_cfg, det_ckpt, device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_estimator = init_pose_estimator(
        pose_cfg, pose_ckpt, device=device,
        cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False)))
    )

    # 🔹 프레임 목록
    frames = sorted(frame_dir.glob("*.jpg"))
    saved = 0

    # 🔹 Batch 처리
    for start in tqdm(range(0, len(frames), batch_size), desc="Sapiens", unit="batch"):
        batch_files = frames[start:start + batch_size]
        batch_imgs_bgr = [cv2.imread(str(f)) for f in batch_files]
        batch_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in batch_imgs_bgr if img is not None]

        if not batch_imgs:
            continue

        try:
            # 사람 검출 (Batch)
            dets = inference_detector(detector, batch_imgs)

            for idx_in_batch, (fpath, img_rgb, det) in enumerate(zip(batch_files, batch_imgs, dets)):
                idx_frame = start + idx_in_batch
                pred = det.pred_instances.cpu().numpy()
                keep = (pred.labels == 0) & (pred.scores > 0.5)
                bbs = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)[keep]
                if len(bbs) == 0:
                    continue
                bbs = bbs[nms(bbs, 0.5), :4]

                # 포즈 추정
                pose_results = inference_topdown(pose_estimator, img_rgb, bbs)
                data_sample = merge_data_samples(pose_results)
                inst = data_sample.get("pred_instances", None)
                if inst is None:
                    continue
                inst_list = split_instances(inst)

                # JSON 저장
                payload = dict(
                    frame_index=idx_frame,
                    meta_info=pose_estimator.dataset_meta,
                    instance_info=inst_list
                )
                json_path = json_dir / f"{idx_frame:06d}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(to_py(payload), f, ensure_ascii=False, indent=2)
                saved += 1

        except Exception as e:
            print(f"[ERROR] batch {start} → {e}")

    return saved
