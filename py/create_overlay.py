import cv2, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_overlay(frame_dir: str, json_dir: str, out_mp4: str, fps: int = 30,
                   kp_radius: int = 4, line_thickness: int = 2):
    """
    프레임 + keypoints JSON → overlay mp4 생성 (COCO 17kp 구조 대응, 모든 keypoints 표시)

    Args:
        frame_dir (str): 프레임 이미지 폴더
        json_dir (str): keypoints JSON 폴더
        out_mp4 (str): 출력 mp4 경로
        fps (int): 초당 프레임 수
        kp_radius (int): keypoint 점 반경
        line_thickness (int): skeleton 선 굵기
    """
    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    if not frame_files:
        print(f"[WARN] No frames found in {frame_dir}")
        return

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)  # 상위 폴더 생성

    # 해상도 확인
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_path in tqdm(frame_files, total=len(frame_files), desc="Creating Overlay", unit="frame"):
        frame = cv2.imread(str(frame_path))
        json_path = Path(json_dir) / (frame_path.stem + ".json")

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "instance_info" in data and len(data["instance_info"]) > 0:
                inst = data["instance_info"][0]  # 첫 번째 인스턴스만 사용
                kpts = np.array(inst["keypoints"])  # (17,2)
                skeleton = data["meta_info"]["skeleton_links"]

                # ---------------- Keypoints (confidence 무시, 모두 표시) ----------------
                for (x, y) in kpts:
                    cv2.circle(frame, (int(x), int(y)), kp_radius, (0, 255, 0), -1)

                # ---------------- Skeleton (무조건 연결) ----------------
                for i, j in skeleton:
                    if i < len(kpts) and j < len(kpts):
                        pt1, pt2 = tuple(map(int, kpts[i])), tuple(map(int, kpts[j]))
                        cv2.line(frame, pt1, pt2, (255, 128, 0), line_thickness)

        writer.write(frame)

    writer.release()