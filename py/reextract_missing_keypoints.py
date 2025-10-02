import cv2, json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mmpose.apis import inference_topdown
from mmpose.structures import merge_data_samples, split_instances

def to_py(obj):
    """넘파이 객체를 JSON 직렬화 가능한 타입으로 변환"""
    import numpy as _np
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, (_np.integer,)):  return int(obj)
    if isinstance(obj, dict):  return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_py(v) for v in obj]
    return obj


def reextract_missing_keypoints(
    file_name: str,              # 비디오 파일명
    frame_dir: str,              # 프레임 디렉토리
    json_dir: str,               # JSON 저장 디렉토리
    n_extracted_frames: int,     # 총 추출된 프레임 수
    pose_estimator,              # 초기화된 Sapiens 포즈 추정 모델
    batch_size: int = 16         # ✅ batch size 옵션
) -> int:
    """
    누락된 프레임만 Sapiens로 재추출 (bbox는 인접 JSON에서 재활용)
    """
    frame_dir, json_dir = Path(frame_dir), Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    expected = {f"{i:06d}" for i in range(n_extracted_frames)}
    existing = {p.stem for p in json_dir.glob("*.json")}
    missing = sorted(expected - existing)

    if not missing:
        print(f"[INFO] {file_name}: 누락된 프레임 없음")
        return len(existing)

    # ✅ batch 단위로 프레임 묶고, 각 프레임별 inference 실행
    for i in tqdm(range(0, len(missing), batch_size),
                  desc=f"{file_name} (re-infer)", unit="batch"):
        batch_frames = missing[i:i+batch_size]

        for fidx_str in batch_frames:
            fidx = int(fidx_str)
            fpath = frame_dir / f"{fidx:06d}.jpg"
            jpath = json_dir / f"{fidx:06d}.json"
            if not fpath.exists() or jpath.exists():
                continue

            img_bgr = cv2.imread(str(fpath))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # -------------------------------
            # bbox 재활용 (좌우 인접 JSON 탐색)
            # -------------------------------
            neighbor, off = None, 1
            while True:
                left = json_dir / f"{fidx-off:06d}.json"
                right = json_dir / f"{fidx+off:06d}.json"
                if left.exists():
                    neighbor = left
                    break
                if right.exists():
                    neighbor = right
                    break
                if (fidx-off) < 0 and (fidx+off) >= n_extracted_frames:
                    break
                off += 1

            if neighbor is None:
                continue

            with open(neighbor, "r", encoding="utf-8") as f:
                nb = json.load(f)
            if not nb.get("instance_info"):
                continue

            bbox = np.array(nb["instance_info"][0]["bbox"], dtype=np.float32).reshape(1, 4)

            # -------------------------------
            # ✅ 프레임 단위 inference 실행
            # -------------------------------
            results = inference_topdown(pose_estimator, img_rgb, bbox)
            data_sample = merge_data_samples(results)
            inst = data_sample.get("pred_instances", None)
            if inst is None:
                continue
            inst_list = split_instances(inst)

            # -------------------------------
            # JSON 저장
            # -------------------------------
            payload = dict(
                frame_index=fidx,
                video_name=file_name,
                meta_info=pose_estimator.dataset_meta,
                instance_info=inst_list,
                source="reextract"
            )
            with open(jpath, "w", encoding="utf-8") as f:
                json.dump(to_py(payload), f, ensure_ascii=False, indent=2)

    # 최종 JSON 개수 다시 세서 반환
    final_json_count = len(list(json_dir.glob("*.json")))
    print(f"[INFO] {file_name}: 최종 JSON 개수 {final_json_count}")
    return final_json_count
