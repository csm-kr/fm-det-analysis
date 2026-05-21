#!/bin/bash
# fm-det dev 컨테이너 entrypoint
# - 호스트에서 주입한 GIT_USER_NAME / GIT_USER_EMAIL 을 컨테이너 git config 에 반영
# - ~/.claude 마운트 권한 점검 (경고만, 차단 안 함)
set -e

# ─── git identity (호스트에서 .env 또는 환경변수로 주입) ───
if [ -n "${GIT_USER_NAME:-}" ]; then
    git config --global user.name "${GIT_USER_NAME}"
fi
if [ -n "${GIT_USER_EMAIL:-}" ]; then
    git config --global user.email "${GIT_USER_EMAIL}"
fi

# ─── ~/.claude 마운트 권한 진단 (사용자 안내용, 차단 안 함) ───
if [ -d "${HOME}/.claude" ] && [ ! -w "${HOME}/.claude" ]; then
    echo "[fm-det entrypoint] WARNING: ${HOME}/.claude exists but not writable." >&2
    echo "[fm-det entrypoint]   호스트 UID/GID 미스매치 가능 — env_docker/Dockerfile 의 HOST_UID/HOST_GID build-arg 확인." >&2
    echo "[fm-det entrypoint]   호스트에서: id -u / id -g 결과를 docker-compose.yml 의 build.args 에 명시." >&2
fi

# ─── GPU 가시성 (디버깅용 한 줄 안내) ───
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "${GPU_COUNT}" = "0" ]; then
        echo "[fm-det entrypoint] WARNING: nvidia-smi 가 GPU 를 인식하지 못함 (runtime 설정 확인)." >&2
    fi
fi

exec "$@"
