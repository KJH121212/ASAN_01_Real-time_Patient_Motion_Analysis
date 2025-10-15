import cv2, json
import numpy as np
from pathlib import Path
from tqdm import tqdm

def create_overlay(frame_dir: str, json_dir: str, out_mp4: str, fps: int = 30,
                   kp_radius: int = 4, line_thickness: int = 2):
    """
    프레임 + keypoints JSON → overlay mp4 생성 (COCO 17kp 구조)
    - 좌우 반전 없음
    - 0~4번 keypoints 제외 (얼굴 제외)
    - L/R 색상 구분
    - 하단 안내문구 표시 ("L: Blue | R: Red")
    """

    frame_files = sorted(Path(frame_dir).glob("*.jpg"))
    if not frame_files:
        print(f"[WARN] No frames found in {frame_dir}")
        return

    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    # 해상도 확인
    sample = cv2.imread(str(frame_files[0]))
    h, w = sample.shape[:2]
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # 색상 설정
    COLOR_SK = (50, 50, 50)   # Skeleton = 짙은 회색
    COLOR_L  = (255, 0, 0)    # 왼쪽 keypoints = 파랑
    COLOR_R  = (0, 0, 255)    # 오른쪽 keypoints = 빨강
    COLOR_NEUTRAL = (0, 255, 0) # 중앙부 keypoints = 초록

    LEFT_POINTS  = [5,7,9,11,13,15]
    RIGHT_POINTS = [6,8,10,12,14,16]

    for frame_path in tqdm(frame_files, total=len(frame_files), desc=f"{Path(frame_dir).name}", unit="frame"):
        frame = cv2.imread(str(frame_path))
        json_path = Path(json_dir) / (frame_path.stem + ".json")

        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "instance_info" in data and len(data["instance_info"]) > 0:
                inst = data["instance_info"][0]
                kpts = np.array(inst["keypoints"])
                skeleton = data.get("meta_info", {}).get("skeleton_links", [])

                # Skeleton
                for i, j in skeleton:
                    if i < len(kpts) and j < len(kpts):
                        if i < 5 or j < 5:
                            continue
                        pt1, pt2 = tuple(map(int, kpts[i])), tuple(map(int, kpts[j]))
                        cv2.line(frame, pt1, pt2, COLOR_SK, line_thickness)

                # Keypoints
                for idx, (x, y) in enumerate(kpts):
                    if idx < 5 or x <= 0 or y <= 0:
                        continue
                    if idx in LEFT_POINTS:
                        color = COLOR_L
                    elif idx in RIGHT_POINTS:
                        color = COLOR_R
                    else:
                        color = COLOR_NEUTRAL
                    cv2.circle(frame, (int(x), int(y)), kp_radius, color, -1)

                # 안내 문구
                legend_text = "L: Blue   |   R: Red"
                cv2.putText(frame, legend_text, (20, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        frame = cv2.flip(frame, 1)  # 좌우 반전 복원
        writer.write(frame)

    writer.release()
    print(f"✅ Overlay 완료 → {out_mp4}")