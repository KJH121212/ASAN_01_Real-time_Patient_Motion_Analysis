import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Outlier 탐지 함수
# ---------------------------
def detect_outliers_distance_last_valid(coords, kp, thr_d=150):
    x = coords[:, kp, 0]
    y = coords[:, kp, 1]
    dist = np.zeros(len(x))
    outliers = []
    x_ref, y_ref = x[0], y[0]
    for t in range(len(x)):
        d = np.sqrt((x[t] - x_ref) ** 2 + (y[t] - y_ref) ** 2)
        dist[t] = d
        if d > thr_d:
            outliers.append(t)
        else:
            x_ref, y_ref = x[t], y[t]
    return x, y, dist, np.array(outliers, dtype=int)

# ---------------------------
# JSON에서 keypoints 불러오기
# ---------------------------
def load_keypoints_from_jsons(json_dir):
    coords_all = []
    for fname in sorted(os.listdir(json_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(json_dir, fname), "r") as f:
            data = json.load(f)
        if "instance_info" not in data or len(data["instance_info"]) == 0:
            continue
        keypoints = np.array(data["instance_info"][0]["keypoints"])
        coords_all.append(keypoints)

    if len(coords_all) == 0:
        return None  # JSON 없음
    return np.stack(coords_all, axis=0)


# ---------------------------
# keypoints outlier plot 저장
# ---------------------------
def plot_keypoints_outliers(coords, save_path, thr_d=150):
    T, J, _ = coords.shape
    kp_range = range(5, J)
    ncols, nrows = 3, len(kp_range)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4*nrows))

    for row, kp in enumerate(kp_range):
        x, y, dist, outliers = detect_outliers_distance_last_valid(coords, kp, thr_d=thr_d)

        ax = axes[row, 0]
        ax.plot(x, color="blue")
        if len(outliers) > 0: ax.scatter(outliers, x[outliers], color="red", marker="x")
        ax.set_title(f"Keypoint {kp} - X"); ax.grid(True, linestyle="--", alpha=0.5)

        ax = axes[row, 1]
        ax.plot(y, color="green")
        if len(outliers) > 0: ax.scatter(outliers, y[outliers], color="red", marker="x")
        ax.set_title(f"Keypoint {kp} - Y"); ax.grid(True, linestyle="--", alpha=0.5)

        ax = axes[row, 2]
        ax.plot(dist, color="brown")
        if len(outliers) > 0: ax.scatter(outliers, dist[outliers], color="red", marker="x")
        ax.axhline(thr_d, color="red", linestyle="--")
        ax.set_title(f"Keypoint {kp} - Distance"); ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


# ---------------------------
# metadata.csv 기반 실행
# ---------------------------
def process_from_metadata(BASE_DIR, thr_d=150):
    meta_path = os.path.join(BASE_DIR, "metadata.csv")
    df = pd.read_csv(meta_path)

    output_root = os.path.join(BASE_DIR, "outlier_plots")
    os.makedirs(output_root, exist_ok=True)

    for _, row in df.iterrows():
        json_dir = row["keypoints_path"]
        if not os.path.isabs(json_dir):
            json_dir = os.path.join(BASE_DIR, json_dir)

        if not os.path.isdir(json_dir):
            print(f"[SKIP] {json_dir} 없음")
            continue

        coords = load_keypoints_from_jsons(json_dir)
        if coords is None:   # JSON 없으면 스킵
            print(f"[SKIP] {json_dir} → JSON 없음")
            continue

        # 상대 경로 계산 (BASE_DIR 기준)
        rel_path = os.path.relpath(json_dir, BASE_DIR)
        parts = rel_path.split(os.sep)

        # 앞쪽 2_KEYPOINTS 제거
        if parts[0] == "2_KEYPOINTS":
            parts = parts[1:]

        # JSON 디렉토리 이름에서 _JSON 제거 → 파일명으로 사용
        json_dir_name = parts[-1]
        filename = json_dir_name.replace("_JSON", "") + ".png"

        # 상위 디렉토리까지만 경로 생성
        save_dir = os.path.join(output_root, *parts[:-1])
        save_path = os.path.join(save_dir, filename)

        plot_keypoints_outliers(coords, save_path, thr_d=thr_d)
        print(f"[DONE] {rel_path} → {save_path}")

# ===========================
# 실행 예시
# ===========================
if __name__ == "__main__":
    BASE_DIR = "/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove/data"
    process_from_metadata(BASE_DIR, thr_d=150)
