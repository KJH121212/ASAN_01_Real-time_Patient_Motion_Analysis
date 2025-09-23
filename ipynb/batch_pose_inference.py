#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CSV에 기록된 여러 영상 프레임 디렉토리를 순서대로 처리하여
Sapiens pose 모델 JSON을 생성하는 배치 스크립트.
- 완료된 파일명은 skip.txt에 기록
- 다음 실행 시 skip.txt에 있는 항목은 건너뜀
"""

import os
import subprocess
import pandas as pd
from pathlib import Path

def main():
    # CSV 파일 경로
    csv_path = "./video_metadata.csv"
    print(f"[DEBUG] CSV 경로: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] CSV 파일 없음: {csv_path}")

    # skip.txt 불러오기
    skip_file = "./skip.txt"
    if os.path.exists(skip_file):
        with open(skip_file, "r") as f:
            skip_list = set(line.strip() for line in f if line.strip())
    else:
        skip_list = set()
    print(f"[DEBUG] skip.txt 불러옴: {len(skip_list)}개")

    # CSV 불러오기
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] CSV 로드 완료. 총 {len(df)}행")

    # frame_count 기준 정렬
    if "frame_count" not in df.columns:
        raise ValueError("CSV에 'frame_count' 컬럼이 없습니다.")
    df = df.sort_values("frame_count")

    # 실행할 스크립트
    SCRIPT_PATH = "../sapiens/pose/scripts/demo/local/keypoints17-py.sh"
    if not os.path.exists(SCRIPT_PATH):
        raise FileNotFoundError(f"[ERROR] 스크립트 없음: {SCRIPT_PATH}")
    print(f"[DEBUG] 실행할 스크립트: {SCRIPT_PATH}")

    # 출력 루트
    OUTPUT_ROOT = "../data/Patient_data/sapiens_output"
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    total = len(df)
    success, skipped, failed = 0, 0, 0

    for idx, row in df.iterrows():
        video_name = row["video_name"]
        frame_dir  = Path(row["frame_dir"])
        frame_count = int(row["frame_count"])

        # skip.txt에 있으면 스킵
        if video_name in skip_list:
            print("=" * 80)
            print(f"[SKIP] ({idx+1}/{total}) {video_name} → skip.txt에 포함됨")
            skipped += 1
            continue

        # 입력 폴더 확인
        if not frame_dir.exists():
            print("=" * 80)
            print(f"[SKIP] ({idx+1}/{total}) {video_name} → 입력 폴더 없음")
            skipped += 1
            continue

        output_dir = Path(OUTPUT_ROOT) / f"{video_name}_JSON"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"[INFO] ({idx+1}/{total}) 처리 시작: {video_name}")

        try:
            subprocess.run([
                "bash", SCRIPT_PATH,
                str(frame_dir),
                str(output_dir)
            ], check=True)
            print(f"[완료] {video_name} JSON 생성 완료\n")
            success += 1

            # ✅ 성공 시 skip.txt에 추가
            with open(skip_file, "a") as f:
                f.write(video_name + "\n")
            skip_list.add(video_name)

        except subprocess.CalledProcessError as e:
            print(f"[실패] {video_name} → 에러 발생 (코드 {e.returncode})\n")
            failed += 1

    print("=" * 80)
    print(f"[요약] 총 {total}개 중 성공 {success}, 스킵 {skipped}, 실패 {failed}")

if __name__ == "__main__":
    main()
