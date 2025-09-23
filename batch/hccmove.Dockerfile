# PyTorch 2.1.0 + CUDA 11.8 (conda 포함)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace

# ===== 시스템 의존성 =====
RUN sed -i 's|http://[a-z]\+.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      git curl wget tzdata ca-certificates \
      build-essential pkg-config \
      libgl1-mesa-glx libglib2.0-0 libgtk2.0-dev \
      ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ===== Locale 설정 (한글/영문 UTF-8) =====
RUN apt-get update && apt-get install -y locales && \
    echo "ko_KR.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen && \
    update-locale LANG=ko_KR.UTF-8 LC_ALL=ko_KR.UTF-8


ENV LANG=ko_KR.UTF-8
ENV LANGUAGE=ko_KR:ko
ENV LC_ALL=ko_KR.UTF-8

# ===== pip 기본 세팅 =====
RUN python -m pip install -U pip setuptools wheel --no-cache-dir

# ===== (1) PyTorch (CUDA 11.8) 1차 =====
RUN python -m pip install --no-cache-dir \
      torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
      --index-url https://download.pytorch.org/whl/cu118

# ===== (2) 과학 연산 스택 =====
RUN python -m pip install --no-cache-dir \
      numpy==1.26.4 scipy==1.11.4 pandas==2.3.2 seaborn==0.13.2

# ===== (3) 유틸 =====
RUN python -m pip install --no-cache-dir \
      pyyaml==6.0.2 tqdm==4.65.2 shapely==2.1.1 json-tricks==3.17.3

# ===== (4) 이미지/비디오 =====
RUN python -m pip install --no-cache-dir \
      opencv-python==4.8.1.78 pillow==11.1.0

# ===== (5) COCO =====
RUN python -m pip install --no-cache-dir \
      Cython==3.1.3 pycocotools==2.0.10 xtcocotools==1.14.3

# ===== (6) pip 툴체인 업 =====
RUN python -m pip install -U pip setuptools wheel \
      --no-cache-dir \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org \
      --trusted-host download.openmmlab.com

# ===== (8) mmengine =====
RUN python -m pip install --no-cache-dir \
      mmengine==0.10.7 \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org

# ===== (9) mmcv (OpenMMLab wheel 링크) =====
RUN python -m pip install --no-cache-dir \
      mmcv==2.1.0 \
      --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html \
      --trusted-host download.openmmlab.com \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org

# ===== (10) mmdet, mmpose =====
RUN python -m pip install --no-cache-dir \
      mmdet==3.2.0 mmpose==1.3.2 \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org

# ===== (옵션) ViT 백본 쓰면 필요: mmpretrain 1.x =====
RUN python -m pip install --no-cache-dir "mmpretrain>=1.0.0,<2.0.0"

# ===== (11) YOLO =====
RUN python -m pip install "ultralytics<9" "deep-sort-realtime==1.3.2"



# ===== 런타임용 헬퍼 스크립트 복사 =====
COPY install_repo_editables.sh /opt/setup/install_repo_editables.sh
RUN chmod +x /opt/setup/install_repo_editables.sh

# Jupyter 편의 (원하면 커널 등록)
RUN python -m pip install --no-cache-dir \
      ipykernel==6.29.5 jupyterlab==4.4.6 notebook==7.4.5

# 커널 등록 (원래 Step 21)
RUN python -m ipykernel install --user --name sapiens --display-name "Python (sapiens)"

# 기본 진입점은 bash
CMD ["/bin/bash"]
