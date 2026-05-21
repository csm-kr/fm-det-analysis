# fm-det — env_docker/ 명령 단축
#
# 호스트에서 `docker compose -f env_docker/docker-compose.yml ...` 을 짧게 치기 위한 alias.
# Makefile 이 싫으면 alias 로 대체:
#   alias dc='docker compose -f env_docker/docker-compose.yml'

COMPOSE := docker compose -f env_docker/docker-compose.yml

.PHONY: up down shell build logs nvidia-test ps restart

up:           ## 컨테이너 띄움 (백그라운드, 빌드 포함)
	$(COMPOSE) up -d --build

down:         ## 컨테이너 종료 + 네트워크 정리 (named volume 은 유지)
	$(COMPOSE) down

shell:        ## dev 컨테이너에 bash 진입
	$(COMPOSE) exec dev bash

build:        ## 이미지만 빌드 (캐시 활용)
	$(COMPOSE) build

logs:         ## dev 컨테이너 로그 추적 (Ctrl+C 로 종료)
	$(COMPOSE) logs -f dev

ps:           ## 컨테이너 상태 확인
	$(COMPOSE) ps

restart:      ## 컨테이너 재시작 (이미지 재빌드 없음)
	$(COMPOSE) restart dev

nvidia-test:  ## GPU 인식 확인 — nvidia-smi 출력
	$(COMPOSE) run --rm dev nvidia-smi

help:         ## 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
