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
      numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 seaborn==0.13.2

# ===== (3) 유틸 =====
RUN python -m pip install --no-cache-dir \
      pyyaml==6.0.2 tqdm==4.66.4 shapely==2.1.1 json-tricks==3.17.3 rich==13.8.0

# ===== (4) 이미지/비디오 처리 =====
RUN python -m pip install --no-cache-dir \
      opencv-python-headless==4.10.0.84 pillow==10.4.0 matplotlib==3.9.2

# ===== (5) pip 툴체인 업그레이드 =====
RUN python -m pip install -U pip setuptools wheel \
      --no-cache-dir \
      --trusted-host pypi.org \
      --trusted-host files.pythonhosted.org

# ===== (6) YOLOv11 + Tracking =====
RUN python -m pip install --no-cache-dir \
      "ultralytics>=8.3.30,<8.4.0" \
      "deep-sort-realtime==1.3.2"

# ===== (7) Torch 유틸 =====
RUN python -m pip install --no-cache-dir \
      torchinfo==1.8.0 torchsummary==1.5.1

# ===== 런타임용 헬퍼 스크립트 복사 =====
COPY install_repo_editables.sh /opt/setup/install_repo_editables.sh
RUN chmod +x /opt/setup/install_repo_editables.sh

# Jupyter 편의 (원하면 커널 등록)
RUN python -m pip install --no-cache-dir \
      ipykernel==6.29.5 jupyterlab==4.4.6 notebook==7.4.5

# 기본 진입점은 bash
CMD ["/bin/bash"]
