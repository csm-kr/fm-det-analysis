# ─────────────────────────────────────────────────
# Stage 1: base — PyTorch 2.7.1 + CUDA 12.8 + cuDNN 9 (devel)
#
# Blackwell (sm_120 — RTX PRO 6000 등) 지원 필수.
# PyTorch 2.7.0 stable 부터 sm_120 공식 지원 (cu128 wheel 에 sm_120 binary 포함).
# 2.6.x / 2.5.x stable wheel 은 sm_50~90 까지만 — Blackwell GPU 에서 "no kernel image" 에러 (I-06).
# wheel 의 supported sm 사전 확인:
#   python -c "import torch; print(torch.cuda.get_arch_list())"
# capability (torch.cuda.get_device_capability) 와 arch_list 의 교집합이 비면 부적합.
#
# 재현성 강화 팁: 아래 태그를 digest 로 핀하면 동일 이미지 보장.
#   docker pull pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
#   docker inspect --format='{{index .RepoDigests 0}}' pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel
# 결과 → 아래 FROM 라인을 그것으로 교체.
# ─────────────────────────────────────────────────
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 외부 의존성 버전 핀 (한 곳에서 관리)
ARG NODE_MAJOR=20
ARG CLAUDE_CODE_VERSION=2.1.123
ARG CLAUDE_PLUGINS_COMMIT=0742692199b49af5c6c33cd68ee674fb2e679d50
# 호스트 UID/GID 미스매치 보정 — 호스트에서 `id -u` / `id -g` 결과로 build-arg 주입
ARG HOST_UID=1000
ARG HOST_GID=1000

# ─────────────────────────────────────────────────
# System dependencies
#   - libgl1, libglib2.0-0: opencv / matplotlib headless 의존
#   - gosu: entrypoint 권한 다운그레이드 (사용 시)
#   - jq: harness 의 success_metric jq 표현 평가 (.claude/skills/harness.md §5-3)
#   - unzip: 데이터셋 zip 해제 (COCO/VOC)
# ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates gosu \
        libgl1 libglib2.0-0 \
        build-essential \
        jq unzip \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS — Claude Code 설치에 필요
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# GitHub CLI (gh) — PR 자동 생성·이슈 조회 등
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gh && \
    rm -rf /var/lib/apt/lists/*

# Claude Code CLI (버전 핀)
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

# ─────────────────────────────────────────────────
# Stage 2: project — Python 의존성 (얇은 레이어, 캐시 최적화)
# ─────────────────────────────────────────────────
FROM base AS project

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# ─────────────────────────────────────────────────
# Stage 3: dev — 작업용 최종 이미지
# ─────────────────────────────────────────────────
FROM project AS dev

# non-root user — 호스트 UID/GID 일치 (bind 마운트 권한 문제 회피)
RUN groupadd -g ${HOST_GID} docker_user 2>/dev/null || true && \
    useradd -m -s /bin/bash -u ${HOST_UID} -g ${HOST_GID} docker_user 2>/dev/null || \
    useradd -m -s /bin/bash docker_user

WORKDIR /workspace/fm-det

# Claude Code 플러그인 marketplace 사전 등록 (commit 핀)
# 호스트 ~/.claude 가 마운트되면 그쪽이 우선이지만, 마운트 없는 경우에도 동작 보장
RUN mkdir -p /home/docker_user/.claude/plugins/marketplaces && \
    git clone https://github.com/anthropics/claude-plugins-official.git \
        /home/docker_user/.claude/plugins/marketplaces/claude-plugins-official && \
    git -C /home/docker_user/.claude/plugins/marketplaces/claude-plugins-official \
        checkout "${CLAUDE_PLUGINS_COMMIT}" && \
    printf '%s\n' \
        '{' \
        '  "claude-plugins-official": {' \
        '    "source": {"source": "github", "repo": "anthropics/claude-plugins-official"},' \
        '    "installLocation": "/home/docker_user/.claude/plugins/marketplaces/claude-plugins-official"' \
        '  }' \
        '}' > /home/docker_user/.claude/plugins/known_marketplaces.json && \
    chown -R docker_user:docker_user /home/docker_user/.claude

# 캐시 디렉터리 미리 생성 (named volume 마운트 전제, 권한 보정)
RUN mkdir -p /home/docker_user/.cache/pip /home/docker_user/.cache/huggingface && \
    chown -R docker_user:docker_user /home/docker_user/.cache

# entrypoint — 호스트 GIT_USER_NAME/EMAIL 주입
COPY env_docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

USER docker_user
ENV HOME=/home/docker_user

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
