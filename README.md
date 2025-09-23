sapiens_lab/
├─ demo/
│  └─ demo_vis.py                    # 데모 실행 스크립트(사람 검출 + 포즈 시각화)
├─ configs/
│  ├─ _base_/
│  │  └─ default_runtime.py          # mmengine 기본 런타임 설정
│  └─ sapiens_pose/
│     └─ coco/
│        └─ sapiens_0.3b-210e_coco-1024x768.py  # 사피엔스 포즈 cfg (COCO)
├─ datasets/
│  └─ coco.py                        # COCO 17kps 메타/헬퍼
├─ scripts/
│  └─ keypoints17.sh                 # 배치 추론(리스트 분할+demo_vis.py 호출)
├─ mmdetection_cfg/
│  └─ rtmdet_person.cfg              # ← 사람 검출기 cfg(예: RTMDet-person) 넣기
├─ checkpoints/
│  ├─ detector/
│  │  └─ rtmdet_person.pth           # ← 사람 검출기 가중치
│  └─ pose/
│     └─ sapiens_0.3b_coco_best.pth  # ← Sapiens-Pose 가중치
├─ data/
│  ├─ images/                        # 실험용 이미지 폴더
│  └─ lists/                         # (선택) 이미지 경로 리스트(txt) 보관
├─ output/
│  └─ vis/                           # 결과 이미지/JSON 출력 위치
└─ notebooks/
   └─ Sapiens_Pose_QuickStart.ipynb  # 노트북(아래에서 바로 받기)
