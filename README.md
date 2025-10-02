# 실시간 환자 동작 분석 (Movement Detection & Analysis)

## 📌 개요
이 프로젝트는 **재활 및 임상 연구를 위한 실시간 환자 동작 분석**을 목표로 합니다.  
비디오를 캡처한 후 프레임을 추출하고, 포즈 추론을 통해 관절 키포인트를 얻어 구조화된 메타데이터를 생성합니다.  
전체 파이프라인은 모듈화·재현성·대규모 확장을 고려하여 설계되었습니다.

---

## 📂 프로젝트 구조

> **설명**  
> - `batch/` : 로그 및 배치 실행 관련 파일  
> - `data/` : 원본 데이터, 프레임, 키포인트, 변환된 mp4, 임시 데이터 등을 포함한 핵심 데이터 디렉토리  
> - `docs/` : 프로젝트 문서  
> - `ipynb/` : 주피터 노트북 파일 및 JSON 교정 결과  
> - `py/` : 파이썬 소스 코드 모듈  
> - `sapiens/` : 외부 pose-estimation 라이브러리 및 서브모듈 (세부 구조는 생략)  

---

## ⚙️ 파이프라인
1. **비디오 전처리**
   - 프레임 추출 (`extract_frames.py`)
   - `ffprobe`를 통한 메타데이터 수집 (fps, 해상도, 코덱, 길이)

2. **포즈 추론**
   - YOLOv8 → 환자 바운딩 박스 추출
   - Sapiens-Pose / RTMDet / MMPose → 키포인트 검출
   - 프레임별 JSON 저장

3. **오버레이 생성**
   - 스켈레톤 오버레이 (`create_overlay.py`)
   - 주석된 MP4 출력

4. **검증 및 무결성 체크**
   - 프레임 수 vs. JSON 개수 비교
   - 메타데이터 일관성 검증
   - 누락된 키포인트 재추출

---

## 🛠️ 주요 스크립트
- `extract_frames.py` – 원본 영상에서 프레임 추출  
- `extract_keypoints.py` – 프레임에 대해 포즈 추론 수행  
- `create_overlay.py` – 영상에 스켈레톤 오버레이 생성  
- `process_video.py` – 엔드 투 엔드 파이프라인 실행  
- `reextract_missing_keypoints.py` – 누락 JSON 처리  
- `sapiens_raw_data.py` – `metadata.csv` 업데이트  

---

## 📑 메타데이터
`metadata.csv`는 모든 영상 처리 과정을 기록하는 테이블입니다:
- `file_name`, `codec`, `width`, `height`, `fps`, `duration`, `nb_frames`
- `frames_verified`, `json_ratio`, `processed_status`
- 처리 및 검증 시점 타임스탬프

---

## 🚀 요구 사항
- batch 폴더 내 hccmove.Dockerfile

---

## 📖 사용법
```bash
# 1. 프레임 추출
python py/extract_frames.py --input RAW_DATA --output FRAME

# 2. 포즈 추론 실행
python py/extract_keypoints.py --input FRAME --output KEYPOINTS

# 3. 오버레이 영상 생성
python py/create_overlay.py --input FRAME --keypoints KEYPOINTS --output MP4
