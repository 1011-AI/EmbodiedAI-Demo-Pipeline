ARG CUDA_BASE_IMAGE=nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${CUDA_BASE_IMAGE} AS os-base

LABEL io.training-platform.dev-compatible="true"

ARG DEBIAN_FRONTEND=noninteractive
ARG MINICONDA_SHA256=ecb43ee4ae30a7a5af87737e9548ceb21f0a10ec55b8dc40d247aa925b80bfec
ARG MINICONDA_URL=https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_26.5.3-1-Linux-x86_64.sh
ARG TARGETARCH
ARG UBUNTU_MIRROR=http://mirrors.aliyun.com/ubuntu
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV CONDA_DIR=/opt/conda \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:${PATH} \
    TZ=Asia/Shanghai \
    PIP_INDEX_URL=${PYPI_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# CUDA 12.8 is already installed in the NVIDIA image. Remove its build-time
# repository before apt update; all additional Ubuntu packages use Aliyun.
RUN rm -f /etc/apt/sources.list.d/cuda*.list \
    && for ubuntu_source in \
         /etc/apt/sources.list \
         /etc/apt/sources.list.d/ubuntu.sources; do \
         if [ -f "${ubuntu_source}" ]; then \
           sed -i \
             -e "s|http://archive.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
             -e "s|http://security.ubuntu.com/ubuntu|${UBUNTU_MIRROR}|g" \
             -e "s|http://ports.ubuntu.com/ubuntu-ports|${UBUNTU_MIRROR}-ports|g" \
             "${ubuntu_source}"; \
         fi; \
       done \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      bash \
      build-essential \
      ca-certificates \
      curl \
      git \
      git-lfs \
      make \
      openssh-server \
      rsync \
      unzip \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && install -d -m 0755 /run/sshd \
    && install -d -m 0700 /root/.ssh \
    && root_password="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')" \
    && printf 'root:%s\n' "${root_password}" | chpasswd \
    && unset root_password \
    && printf '%s\n' \
      'PermitRootLogin prohibit-password' \
      'PubkeyAuthentication yes' \
      'PasswordAuthentication no' \
      > /etc/ssh/sshd_config.d/10-training-platform-defaults.conf \
    && /usr/sbin/sshd -t \
    && if [ "${TARGETARCH:-amd64}" != "amd64" ]; then \
         echo "This image currently supports linux/amd64 only" >&2; exit 2; \
       fi \
    && curl -fL --retry 5 -o /tmp/miniconda.sh "${MINICONDA_URL}" \
    && echo "${MINICONDA_SHA256}  /tmp/miniconda.sh" | sha256sum -c - \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm -f /tmp/miniconda.sh \
    && printf '%s\n' \
      'channels:' \
      '  - defaults' \
      'show_channel_urls: true' \
      'default_channels:' \
      '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' \
      '  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' \
      'custom_channels:' \
      '  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud' \
      > "${CONDA_DIR}/.condarc" \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version'

RUN sed -i \
      's|^Subsystem[[:space:]][[:space:]]*sftp[[:space:]].*$|Subsystem sftp internal-sftp|' \
      /etc/ssh/sshd_config \
    && /usr/sbin/sshd -t

COPY --chmod=0644 <<'EOF' /etc/embodied-ai/paths.txt
EmbodiedAI container paths

  Preinstalled upstream dependency sources:
    /opt/embodied-ai-dependencies

  Python virtual environments:
    /opt/venvs

  Platform-mounted user home and shared workspace:
    /workspace

The image does not include the EmbodiedAI Demo Pipeline source and does not
predefine locations for project clones, runtime data, models, caches, or
outputs. /workspace is not created by the image and remains available as a
platform mount point. Set PROJECT_ROOT and storage/cache environment variables
only when the project or a workload needs them.
EOF

COPY --chmod=0755 <<'EOF' /etc/profile.d/embodied-ai-paths.sh
#!/bin/sh

case "$-" in
  *i*)
    if [ -z "${EMBODIED_AI_PATHS_SHOWN:-}" ] && [ -r /etc/embodied-ai/paths.txt ]; then
      EMBODIED_AI_PATHS_SHOWN=1
      cat /etc/embodied-ai/paths.txt
    fi
    ;;
esac
EOF

RUN printf '%s\n' \
      '. /opt/conda/etc/profile.d/conda.sh' \
      '. /etc/profile.d/embodied-ai-paths.sh' \
      >> /etc/bash.bashrc

WORKDIR /root

FROM os-base AS core

RUN mkdir -p /opt/embodied-ai-dependencies /opt/venvs \
    && python3 -m venv /opt/venvs/core \
    && /opt/venvs/core/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venvs/core/bin/python -m pip install \
      "pydantic>=2.8,<3" \
      "PyYAML>=6,<7" \
      "pytest>=8,<9" \
    && /opt/venvs/core/bin/python -c "import pydantic, pytest, yaml"

FROM core AS cuda-base

ARG CUDA_PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG PYTORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cu128/

# Media, rendering and compiler packages are needed only by the GPU research
# environments. They are still Ubuntu packages and use the configured mirror.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      cmake \
      ffmpeg \
      g++ \
      gcc \
      libegl1 \
      libexpat1 \
      libfontconfig1-dev \
      libgl1 \
      libglib2.0-0 \
      libglfw3 \
      libglx-mesa0 \
      libmagickwand-dev \
      libosmesa6-dev \
      libsm6 \
      libsndfile1 \
      libvulkan1 \
      libxext6 \
      libxrender1 \
      mesa-vulkan-drivers \
      ninja-build \
      pkg-config \
      sox \
      vulkan-tools \
    && rm -rf /var/lib/apt/lists/*

ENV CC=/usr/bin/gcc \
    CUDA_HOME=/usr/local/cuda-12.8 \
    CXX=/usr/bin/g++ \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all

# Install one shared CUDA-enabled PyTorch stack. GPU virtual environments use
# system-site-packages so the multi-gigabyte CUDA wheels are not duplicated.
# Generic and NVIDIA dependencies come from the configured domestic PyPI mirror;
# PyTorch's cu128 wheels come from Aliyun's domestic PyTorch wheel mirror.
RUN python3 -m pip install --break-system-packages \
      --index-url "${CUDA_PYPI_INDEX_URL}" uv \
    && uv pip install --system --break-system-packages --no-cache \
      --index-url "${CUDA_PYPI_INDEX_URL}" \
      filelock \
      fsspec \
      jinja2 \
      networkx \
      numpy \
      "nvidia-cublas-cu12==12.8.3.14" \
      "nvidia-cuda-cupti-cu12==12.8.57" \
      "nvidia-cuda-nvrtc-cu12==12.8.61" \
      "nvidia-cuda-runtime-cu12==12.8.57" \
      "nvidia-cudnn-cu12==9.7.1.26" \
      "nvidia-cufft-cu12==11.3.3.41" \
      "nvidia-cufile-cu12==1.13.0.11" \
      "nvidia-curand-cu12==10.3.9.55" \
      "nvidia-cusolver-cu12==11.7.2.55" \
      "nvidia-cusparse-cu12==12.5.7.53" \
      "nvidia-cusparselt-cu12==0.6.3" \
      "nvidia-nccl-cu12==2.26.2" \
      "nvidia-nvjitlink-cu12==12.8.61" \
      "nvidia-nvtx-cu12==12.8.55" \
      pillow \
      "sympy>=1.13.3" \
      "triton==3.3.1" \
      "typing-extensions>=4.10.0" \
    && uv pip install --system --break-system-packages --no-cache --no-deps \
      --no-index --find-links "${PYTORCH_FIND_LINKS}" \
      "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" \
    && python3 - <<'PY'
import torch
import torchvision

assert torch.version.cuda == "12.8", torch.version.cuda
print(f"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda}")
PY

FROM cuda-base AS lerobot

ARG GPU_PYPI_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
ARG GPU_PYPI_FALLBACK_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG LEROBOT_REF=e40b58a8dfa9e7b86918c374791599d070518d11
ENV PIP_INDEX_URL=${GPU_PYPI_INDEX_URL}

RUN python3 -m venv --system-site-packages /opt/venvs/lerobot \
    && /opt/venvs/lerobot/bin/python -m pip install --upgrade pip setuptools wheel

RUN --mount=type=bind,source=scripts/lerobot/install_lerobot_cluster.sh,target=/tmp/install_lerobot_cluster.sh,ro \
    /opt/venvs/lerobot/bin/python -m pip install \
      --index-url "${GPU_PYPI_FALLBACK_URL}" "num2words==0.5.14" \
    && PATH=/opt/venvs/lerobot/bin:${PATH} \
    LEROBOT_REF=${LEROBOT_REF} \
    LEROBOT_SOURCE_DIR=/opt/embodied-ai-dependencies/lerobot \
    LEROBOT_CREATE_CONDA=0 \
    LEROBOT_INSTALL_NO_DEPS=1 \
    LEROBOT_FORCE_OPENCV_HEADLESS=1 \
    LEROBOT_SKIP_TORCH_INSTALL=1 \
    LEROBOT_SKIP_PIP_BOOTSTRAP=1 \
    bash /tmp/install_lerobot_cluster.sh

FROM lerobot AS runtime

ARG IMAGEWAM_REPO=https://github.com/yuyangalin/ImageWAM.git
ARG IMAGEWAM_REF=70bc47147a49652fc3ca62617cab0e0b6b31bc92
ARG FLUX2_REPO=https://github.com/black-forest-labs/flux2.git
ARG FLUX2_REF=50fe5162777813d869182b139e83b10743caef15
ARG LIBERO_REPO=https://github.com/Lifelong-Robot-Learning/LIBERO.git
ARG LIBERO_REF=8f1084e3132a39270c3a13ebe37270a43ece2a01

RUN git clone "${IMAGEWAM_REPO}" /opt/embodied-ai-dependencies/ImageWAM \
    && git -C /opt/embodied-ai-dependencies/ImageWAM checkout --detach "${IMAGEWAM_REF}" \
    && git clone "${FLUX2_REPO}" /opt/embodied-ai-dependencies/ImageWAM/third_party/flux2 \
    && git -C /opt/embodied-ai-dependencies/ImageWAM/third_party/flux2 checkout --detach "${FLUX2_REF}" \
    && git clone "${LIBERO_REPO}" /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO \
    && git -C /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO checkout --detach "${LIBERO_REF}" \
    && touch /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero/__init__.py \
    && sed -i \
      's/torch.load(init_states_path)/torch.load(init_states_path, weights_only=False)/' \
      /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero/benchmark/__init__.py

RUN python3 -m venv --system-site-packages /opt/venvs/imagewam \
    && /opt/venvs/imagewam/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venvs/imagewam/bin/python -m pip install \
      -e "/opt/embodied-ai-dependencies/ImageWAM[shared]" \
      "mujoco==3.3.2" \
      "robosuite==1.4.0" \
      "bddl==1.0.1" \
      "gym==0.25.2" \
      "diffusers==0.34.0" \
      "opencv-python==4.11.0.86" \
      "opencv-python-headless==4.11.0.86" \
      "scipy==1.17.1" \
      easydict thop future cloudpickle \
    && /opt/venvs/imagewam/bin/python -m pip install --no-deps \
      -e /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO \
    && mkdir -p /opt/libero-config \
    && printf '%s\n' \
      'benchmark_root: /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero' \
      'bddl_files: /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero/bddl_files' \
      'init_states: /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero/init_files' \
      "datasets: ''" \
      'assets: /opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO/libero/libero/assets' \
      > /opt/libero-config/config.yaml

RUN /opt/venvs/core/bin/python -c "import pydantic, pytest, yaml" \
    && /opt/venvs/lerobot/bin/python -c "import diffusers, lerobot, lerobot.scripts.lerobot_train" \
    && LIBERO_CONFIG_PATH=/opt/libero-config \
      PYTHONPATH=/opt/embodied-ai-dependencies/ImageWAM/third_party/LIBERO \
      /opt/venvs/imagewam/bin/python -c "import imagewam, libero.libero"

ENV DEPENDENCY_ROOT=/opt/embodied-ai-dependencies \
    HF_ENDPOINT=https://hf-mirror.com \
    LIBERO_CONFIG_PATH=/opt/libero-config \
    LEROBOT_SOURCE_DIR=/opt/embodied-ai-dependencies/lerobot \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa

CMD ["bash"]
