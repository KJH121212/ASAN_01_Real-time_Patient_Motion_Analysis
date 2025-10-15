import json                                            # JSON 입출력을 위해 json 모듈을 임포트합니다
import math                                            # 수치 계산을 위해 math 모듈을 임포트합니다
import shutil                                          # 필요 시 파일 복사를 위해 shutil 모듈을 임포트합니다
from pathlib import Path                                # 경로 처리를 위해 Path 클래스를 임포트합니다
from typing import List, Tuple, Dict                    # 타입 힌트를 위해 typing을 임포트합니다

import cv2                                             # 프레임 오버레이 및 mp4 생성을 위해 OpenCV를 임포트합니다
import numpy as np                                     # 배열 연산을 위해 numpy를 임포트합니다
import pandas as pd                                    # metadata.csv 처리를 위해 pandas를 임포트합니다
from tqdm.auto import tqdm                              # 진행 표시를 위해 tqdm.auto에서 tqdm를 임포트합니다

# ====== 고정 경로 설정 ======
BASE_DIR = Path("/workspace/nas203/ds_RehabilitationMedicineData/IDs/Kimjihoo/3_project_HCCmove")  # 프로젝트 기본 경로를 정의합니다
DATA_DIR = BASE_DIR / "data"                                                                        # 데이터 폴더 경로를 정의합니다
CSV_PATH  = DATA_DIR / "metadata.csv"                                                               # 메타데이터 CSV 경로를 정의합니다

# ====== 유틸 함수: 자연 정렬 키 ======
def _natural_key(s: str) -> List:                                                                   # 자연 정렬을 위한 키를 생성하는 함수입니다
    import re                                                                                       # 정규식 처리를 위해 re 모듈을 임포트합니다
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]                    # 숫자와 문자를 분리해 정렬 키를 반환합니다

# ====== 유틸 함수: 로버스트 z-score(MAD) ======
def robust_z(x: np.ndarray) -> np.ndarray:                                                          # MAD 기반 로버스트 z-score를 계산하는 함수입니다
    med   = np.nanmedian(x)                                                                         # 데이터의 중앙값을 계산합니다
    mad   = np.nanmedian(np.abs(x - med))                                                           # 중앙값으로부터의 절대편차의 중앙값을 계산합니다
    scale = 1.4826 * mad if mad > 0 else (np.nanstd(x) if np.nanstd(x) > 0 else 1.0)               # MAD가 0이면 표준편차 또는 1.0으로 대체합니다
    return (x - med) / scale                                                                        # 스케일로 나누어 로버스트 z-score를 반환합니다

# ====== 유틸 함수: LOCF 보간 ======
def interpolate_locf(series: np.ndarray, is_outlier: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # LOCF 보간과 플래그를 반환하는 함수입니다
    corrected = series.copy()                                                                        # 원본을 복사해 보정 배열을 준비합니다
    flags     = np.zeros_like(series, dtype=bool)                                                    # 보간 여부를 기록할 불리언 배열을 생성합니다
    last_good = np.nan                                                                               # 마지막 정상값을 NaN으로 초기화합니다
    for t in range(len(series)):                                                                     # 시계열을 시간 순서로 순회합니다
        if is_outlier[t] or np.isnan(series[t]):                                                     # 아웃라이어이거나 결측이면 보간 로직을 적용합니다
            if not np.isnan(last_good):                                                              # 이전 정상값이 존재하면 LOCF를 수행합니다
                corrected[t] = last_good                                                             # 이전 정상값을 현재 값에 복사합니다
                flags[t]    = True                                                                   # 해당 시점이 보간되었음을 표시합니다
            else:                                                                                    # 초반부로 이전 정상값이 없는 경우를 처리합니다
                corrected[t] = series[t]                                                             # 값 변경 없이 원본을 유지합니다
        else:                                                                                        # 현재 관찰이 정상인 경우를 처리합니다
            last_good = series[t]                                                                    # 마지막 정상값을 현재 값으로 갱신합니다
    return corrected, flags                                                                          # 보정된 시계열과 보간 플래그를 반환합니다

# ====== 유틸 함수: 링크/색상 추출 ======
def get_coco_links(meta_info: Dict) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int, int]]]:    # 스켈레톤 링크와 색상을 반환하는 함수입니다
    links  = meta_info.get("skeleton_links", [])                                                     # 링크 인덱스 페어 목록을 읽습니다
    colors = meta_info.get("skeleton_link_colors", [])                                               # 링크 색상 목록을 읽습니다
    return [tuple(map(int, e)) for e in links], [tuple(map(int, c)) for c in colors]                 # 정수 튜플 리스트로 변환해 반환합니다

# ====== 유틸 함수: 프레임 이미지 목록 ======
def list_frame_images(frame_dir: Path) -> List[Path]:                                                # 프레임 이미지 경로 목록을 반환하는 함수입니다
    exts  = (".jpg", ".jpeg", ".png", ".bmp")                                                        # 지원하는 이미지 확장자를 정의합니다
    files = [p for p in frame_dir.rglob("*") if p.suffix.lower() in exts]                            # 재귀적으로 모든 이미지 파일을 수집합니다
    files.sort(key=lambda p: _natural_key(p.name))                                                   # 자연 정렬 키로 파일 목록을 정렬합니다
    return files                                                                                     # 정렬된 이미지 경로 리스트를 반환합니다

# ====== 비디오 한 건 처리 함수 ======
def process_one_row(row: pd.Series) -> None:
    """metadata.csv의 한 행(row)을 입력받아 보정 및 오버레이 비디오를 생성"""
    # ====== 기본 경로 설정 ======
    fps           = float(row["fps"]) if "fps" in row and not pd.isna(row["fps"]) else 30.0
    video_rel     = Path(str(row.get("video_path", "")).strip())
    keypoints_rel = Path(str(row.get("keypoints_path", "")).strip())
    frame_rel     = Path(str(row.get("frame_path", "")).strip())

    video_path     = (DATA_DIR / video_rel).resolve()
    keypoints_path = (DATA_DIR / keypoints_rel).resolve()
    frame_path     = (DATA_DIR / frame_rel).resolve()

    rel_after_raw_str = str(video_rel).split("0_RAW_DATA/")[-1]
    rel_after_raw     = Path(rel_after_raw_str)

    json_dir_rel = rel_after_raw.with_suffix("")
    OUT_JSON_DIR = DATA_DIR / "4_INTERP_DATA" / "JSON" / json_dir_rel
    OUT_JSON_DIR.mkdir(parents=True, exist_ok=True)

    mp4_dir_rel  = rel_after_raw.parent
    OUT_MP4_DIR  = DATA_DIR / "4_INTERP_DATA" / "MP4" / mp4_dir_rel
    OUT_MP4_DIR.mkdir(parents=True, exist_ok=True)
    out_mp4_path = OUT_MP4_DIR / f"{video_rel.stem}.mp4"

    # ====== 입력 파일 확인 ======
    if not keypoints_path.exists():
        raise FileNotFoundError(f"키포인트 폴더가 없습니다: {keypoints_path}")
    if not frame_path.exists():
        raise FileNotFoundError(f"프레임 폴더가 없습니다: {frame_path}")

    json_files = sorted(keypoints_path.glob("*.json"), key=lambda p: _natural_key(p.name))
    if len(json_files) == 0:
        raise FileNotFoundError(f"JSON 파일이 없습니다: {keypoints_path}")

    # ====== 키포인트 로드 ======
    all_keypoints, all_scores, metas = [], [], None
    for jf in tqdm(json_files, desc="Load JSON", unit="file"):
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        inst = data["instance_info"][0]
        kps  = np.array(inst["keypoints"], dtype=np.float32)
        scs  = np.array(inst.get("keypoint_scores", [np.nan] * len(kps)), dtype=np.float32)
        all_keypoints.append(kps)
        all_scores.append(scs)
        if metas is None:
            metas = data.get("meta_info", {})

    K = np.stack(all_keypoints, axis=0)   # (T,17,2)
    C = np.stack(all_scores, axis=0)      # (T,17)
    Tlen = K.shape[0]
    joint_range = list(range(5, 17))

    # ====== Adaptive Velocity-based Outlier 보정 ======
    def velocity_correct(x, y):
        """프레임간 속도 기반으로 튐(outlier)을 보정"""
        v = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
        v_std, v_max = np.nanstd(v), np.nanmax(v)
        v_mean, v_med = np.nanmean(v), np.nanmedian(v)

        # Outlier가 거의 없는 경우 (정상 동작)
        if v_std < 5 or v_max < 30:
            thr_vel = 9999.0
        else:
            thr_perc = np.nanpercentile(v, 95) * 1.2
            thr_mean = v_mean * 3.0
            thr_vel  = max(thr_perc, thr_mean, 80.0)

            out_mask = v > thr_vel
            if out_mask.mean() > 0.05:    # 과탐 방지
                thr_vel *= 1.5

        # 보정 루프 (실시간 방식)
        x_corr = x.copy()
        y_corr = y.copy()
        vel_prev_x, vel_prev_y = 0.0, 0.0
        for t in range(1, len(x)):
            dx, dy = x_corr[t] - x_corr[t-1], y_corr[t] - y_corr[t-1]
            vel = np.hypot(dx, dy)
            if vel > thr_vel:
                x_corr[t] = x_corr[t-1] + vel_prev_x
                y_corr[t] = y_corr[t-1] + vel_prev_y
            else:
                vel_prev_x, vel_prev_y = dx, dy
        return x_corr, y_corr, thr_vel

    thr_dict = {}
    for j in tqdm(joint_range, desc="Velocity correction", unit="joint"):
        x = K[:, j, 0].astype(float)
        y = K[:, j, 1].astype(float)
        x_corr, y_corr, thr_used = velocity_correct(x, y)
        K[:, j, 0], K[:, j, 1] = x_corr, y_corr
        thr_dict[j] = thr_used

    # ====== 보정된 JSON 저장 ======
    for t, jf in tqdm(list(enumerate(json_files)), total=Tlen, desc="Save JSON", unit="frame"):
        with open(jf, "r", encoding="utf-8") as f:
            data_in = json.load(f)
        inst_in  = data_in["instance_info"][0]
        kps_out  = []
        scs_out  = []
        for new_i, j in enumerate(joint_range):
            kps_out.append([float(K[t, j, 0]), float(K[t, j, 1])])
            if "keypoint_scores" in inst_in:
                scs_out.append(float(C[t, j]))
        meta_out = dict(data_in.get("meta_info", {}))
        links_in, colors_in = get_coco_links(meta_out)
        keep_idx   = [i for i, (a, b) in enumerate(links_in) if a >= 5 and b >= 5]
        links_new  = [(a - 5, b - 5) for i, (a, b) in enumerate(links_in) if i in keep_idx]
        colors_new = [colors_in[i] for i in keep_idx] if colors_in else []
        meta_out["skeleton_links"]       = links_new
        meta_out["skeleton_link_colors"] = colors_new
        meta_out["kept_joint_indices"]   = list(range(5, 17))
        meta_out["joint_index_offset"]   = 5
        data_out = {
            "meta_info": meta_out,
            "instance_info": [{
                "keypoints": kps_out,
                **({"keypoint_scores": scs_out} if scs_out else {}),
            }]
        }
        out_path = OUT_JSON_DIR / jf.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data_out, f, ensure_ascii=False, indent=2)

    # ====== 프레임 오버레이 비디오 생성 ======
    frame_files = list_frame_images(frame_path)
    if len(frame_files) != Tlen:
        print(f"⚠️ 프레임 수({len(frame_files)})와 JSON 수({Tlen})가 다릅니다")

    first_img = cv2.imread(str(frame_files[0]))
    if first_img is None:
        raise RuntimeError(f"프레임 이미지를 읽지 못했습니다: {frame_files[0]}")
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_mp4_path), fourcc, fps, (w, h))

    links_all, link_colors = get_coco_links(metas or {})
    keep_idx_vis = [i for i, (a, b) in enumerate(links_all) if a >= 5 and b >= 5]
    links_vis = [(a, b) for i, (a, b) in enumerate(links_all) if i in keep_idx_vis]
    colors_vis = [link_colors[i] for i in keep_idx_vis] if link_colors else []

    for idx in tqdm(range(Tlen), desc="Render MP4", unit="frame"):
        img = cv2.imread(str(frame_files[idx])) if idx < len(frame_files) else np.zeros((h, w, 3), dtype=np.uint8)
        kps = K[idx]
        for j in range(5, 17):
            x_i, y_i = int(round(kps[j, 0])), int(round(kps[j, 1]))
            cv2.circle(img, (x_i, y_i), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        for (a, b), col in zip(links_vis, colors_vis if colors_vis else [(51, 153, 255)] * len(links_vis)):
            xa, ya = int(round(kps[a, 0])), int(round(kps[a, 1]))
            xb, yb = int(round(kps[b, 0])), int(round(kps[b, 1]))
            cv2.line(img, (xa, ya), (xb, yb),
                     (int(col[2]), int(col[1]), int(col[0])),
                     2, lineType=cv2.LINE_AA)
        writer.write(img)

    writer.release()


def main():                                                                                 # N01 내부 파일들만 처리하는 메인 함수입니다
    meta = pd.read_csv(CSV_PATH)                                                            # metadata.csv를 로드합니다
    if meta.empty:                                                                          # CSV가 비어있는지 확인합니다
        raise RuntimeError("metadata.csv가 비어 있습니다")                                    # 비어 있으면 예외를 발생시킵니다

    # 'AI_dataset/N01' 이 포함된 행만 필터링합니다
    subset = meta[meta["video_path"].astype(str).str.contains("AI_dataset/N01", case=False, na=False)]

    if subset.empty:                                                                        # 해당 경로를 가진 행이 없으면 예외를 발생시킵니다
        raise RuntimeError("metadata.csv에서 'AI_dataset/N01' 경로를 포함한 비디오를 찾지 못했습니다")

    total = len(subset)                                                                     # 전체 처리 대상 개수를 계산합니다
    print(f"🎯 총 {total}개의 비디오가 선택되었습니다 (AI_dataset/N01)")                     # 선택된 비디오 개수를 출력합니다

    # 순차적으로 각 비디오를 처리합니다
    for i, (_, row) in enumerate(subset.iterrows(), start=1):                               # 각 행을 순회하면서 처리 인덱스를 표시합니다
        print(f"\n▶ ({i}/{total}) 처리 중: {row['video_path']}")                            # 현재 진행 상황을 출력합니다
        try:
            process_one_row(row)                                                            # 비디오 한 건 처리 함수를 호출합니다
        except Exception as e:                                                              # 처리 중 오류가 발생한 경우를 처리합니다
            print(f"❌ 오류 발생 ({row['video_path']}): {e}")                               # 오류 메시지를 출력합니다
            continue                                                                        # 다음 비디오로 넘어갑니다


# ====== 실행부 ======
if __name__ == "__main__":                                                                            # 스크립트가 직접 실행될 때만 동작하도록 합니다
    main()                                                                                            # 메인 함수를 호출해 전체 파이프라인을 실행합니다
