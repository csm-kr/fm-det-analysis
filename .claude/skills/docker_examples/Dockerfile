# ────────────────────────────────────────────────
# Base: Ubuntu 22.04 + Python 3.10
#
# 재현성 강화 팁: 아래 태그를 digest로 핀하면 동일 이미지를 보장한다.
#   docker pull ubuntu:22.04
#   docker inspect --format='{{index .RepoDigests 0}}' ubuntu:22.04
# 결과 예: ubuntu:22.04@sha256:<digest>  →  아래 FROM 라인을 그것으로 교체.
# ────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 핀닝된 외부 의존성 버전 (한 곳에서 관리)
ARG NODE_MAJOR=20
ARG CLAUDE_CODE_VERSION=2.1.123
ARG CLAUDE_PLUGINS_COMMIT=0742692199b49af5c6c33cd68ee674fb2e679d50

# ────────────────────────────────────────────────
# System dependencies
# ────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    git wget curl gosu \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS — Claude Code 설치에 필요
RUN curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────
# GitHub CLI (gh) — PR 자동 생성 등에 사용
# ────────────────────────────────────────────────
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && \
    apt-get install -y gh && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────
# Claude Code CLI (버전 핀)
# ────────────────────────────────────────────────
RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# ────────────────────────────────────────────────
# Python dependencies
# ────────────────────────────────────────────────
RUN pip install --upgrade pip && \
    pip install pytest

# ────────────────────────────────────────────────
# Project setup
# ────────────────────────────────────────────────
WORKDIR /workspace/harness_framework

COPY . .

RUN chmod +x scripts/*.sh

# ────────────────────────────────────────────────
# Non-root user
# ────────────────────────────────────────────────
RUN useradd -m -s /bin/bash docker_user && \
    chown -R docker_user:docker_user /workspace/harness_framework

USER docker_user
ENV HOME=/home/docker_user

# ────────────────────────────────────────────────
# Claude Code 플러그인 marketplace 사전 등록 (commit 핀)
# (호스트 ~/.claude 마운트 없이도 컨테이너 안에서 /plugin 사용 가능)
# ────────────────────────────────────────────────
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
        '}' > /home/docker_user/.claude/plugins/known_marketplaces.json

# ────────────────────────────────────────────────
# entrypoint — 호스트에서 주입한 GIT_USER_NAME/GIT_USER_EMAIL을
# 컨테이너 안 git config에 반영한 뒤 CMD를 실행한다.
# ────────────────────────────────────────────────
COPY --chown=docker_user:docker_user scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
USER root
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
USER docker_user

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
