# Pensieve — fm-det 의 지금 상태 한 페이지

> **이 문서가 답하는 질문**: 오랜만에 돌아왔을 때 *지금 어디까지 왔고, 다음에 무엇을 해야 하는가*? 본 문서는 **시점 의존 스냅샷** — 다른 문서들(PRD/ARCH/ADR/EXPERIMENTS/ISSUE) 가 *누적·정책* 인 반면, pensieve 는 *덮어쓰는 한 페이지*. 한 화면에서 컨텍스트가 회복되도록 짧게 유지.

> **갱신 규칙**: 매 작업 종료 시 본 문서를 갱신한다. **"마지막 업데이트" 한 줄·"지금 어디"·"다음 한 가지" 세 곳은 반드시 최신**. 나머지는 변경된 경우에만.

---

## 마지막 업데이트
- **일시**: 2026-05-21
- **갱신자**: Claude (ISSUE/PENSIEVE → docs/ 이동 + M0/M1 마일스톤 도입 + datasets/coco 모듈 정리)

---

## 지금 어디 (현재 단계)
- **전체 단계**: 첫 phase (`data-sanity-coco`) 완료 ✅ + 인프라 정리 완료 (pyproject/.gitignore/Dockerfile-jq 패치). 다음 = train2017 다운로드 + VOC.
- **활성 phase**: 없음 — 다음 phase (`data-sanity-coco-train`) 설계 직전.
- **활성 작업**: 없음 (인프라 정리 결과 보고 중).

## 다음 한 가지 (Single Next Action)
> 막연한 "이것저것" 대신 **다음에 손댈 한 가지**를 적는다. 끝나면 다음 한 가지로 갱신.

**`data-sanity-coco-train` phase 설계 + train2017 (18GB) background 다운로드 시작.** train 받아지는 동안 `data-sanity-voc` phase 도 병행 설계 + 다운로드. 둘 다 다운 완료 후 분석/시각화/리포트 자동 진행.

이후 순서 (참고만):
1. ~~`/docker-init`~~ ✅ (R-03)
2. ~~`data-sanity-coco` val~~ ✅ (CP-1 approved)
3. ~~인프라 정리 (pyproject / .gitignore / Dockerfile jq)~~ ✅ (R-05 / R-06 / I-05 blocked-rebuild)
4. **(다음)** `data-sanity-coco-train` — train2017 18GB 다운 + 분포 분석 (delta vs val)
5. `data-sanity-voc` — VOC 07 + 12 다운 + XML 파싱 + 분포·시각화
6. `/harness` 로 **P0** `coco-repro-baseline` phase 설계
7. P0 학습 → COCO val AP 46.2 ± 0.5 매칭 (I-04)
8. **P0a 메커니즘 진단 5행** — `coco-diag-signal-scale / box-renewal / iter-step / num-boxes / nms-iou`
9. P0a 5행 OK → P1 `coco-fm1-sampler-cfm` 시작

---

## 최근 변경 (최근 5개, 시간 역순)
- **2026-05-21** — **ISSUE/PENSIEVE docs/ 이동 + M0/M1 마일스톤 도입**: `ISSUE.md` → `docs/ISSUE.md` / `pensieve.md` → `docs/PENSIEVE.md` (대문자 + docs/ 일관성). CLAUDE.md 에 "## 마일스톤" 섹션 신설 + M0 (부트스트래핑/docs/컨테이너) + M1 (Pre-P0 데이터 sanity + 인프라). EXPERIMENTS.md 의 단계적 FM 전환 로드맵 표에 `Pre-P0 데이터 sanity` 행 추가. 모든 docs/phases 의 ISSUE/PENSIEVE 참조 surgical 갱신.
- **2026-05-21** — **datasets/coco/ 모듈 정리**: 루트의 `data_{download,sanity,visualize,report}.py` 4개 → `datasets/coco/{download,sanity,visualize,report}.py` 로 이동 (ARCHITECTURE.md 의 datasets/ 정책 일치). 호출 방식 `python -m datasets.coco.<name>`. ARCHITECTURE.md datasets/ 트리 갱신 + DATA_CARD/phases step.md 4개의 참조 surgical 갱신. VOC 도 동일 패턴 (`datasets/voc/`) 으로 도입 예정.
- **2026-05-21** — **인프라 정리**: `pyproject.toml` 신설 (R-05 resolved) + `.gitignore` ai-ml 보강 (R-06 resolved — `data/runs/wandb/outputs/.hydra/*.pt/.pth/.ckpt/.safetensors/__pycache__/...`) + `env_docker/Dockerfile` 에 `jq unzip` apt 추가 (I-05 blocked-rebuild — `make build` 또는 `docker exec -u root ...` 필요). CP-1 approved → `data-sanity-coco` phase completed.
- **2026-05-21** — **`data-sanity-coco` phase 첫 실행**: `data_download.py`/`data_sanity.py`/`data_visualize.py`/`data_report.py` 신설 + COCO val 5000장 + annotations 6 파일 (integrity OK, 1.02GB) 다운 + 분포·박스·해상도 통계 + 4 figure (class_dist / bbox_size / image_size / samples) + report.md 묶음. DATA_CARD 일치 3/3. I-05 (jq 미설치) 신규 등록.
- **2026-05-21** — **`/docker-init` 완료**: `env_docker/{Dockerfile, docker-compose.yml, docker-entrypoint.sh, .dockerignore}` + `Makefile` + `.env.example` + `requirements.txt` 생성. 베이스 = `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, GPU runtime, shm 8gb, TB 6007:6006, `~/.claude` 마운트 (인계). I-02 → R-03 resolved.

## 진행 중 phase

| phase tag | 상태 | step 진행 | runs/ |
|-----------|------|----------|-------|
| `data-sanity-coco` | **completed** (CP-1 approved) | 0,1,2,3 ✅ | `data-sanity-{download,analyze,vis,report}-20260521-{1332,1335}` |

## 최근 ablation / 진단 결과 (최근 3개)
없음 — P0/P0a 미시작. EXPERIMENTS.md 의 진단 표 + ablation 표가 채워지기 시작하면 본 섹션에 최근 3개 행 미러링.

| run_id | dataset | phase | 변경 1변수 | AP / mAP@.5 | ΔAP | 결론 |
|--------|---------|-------|-----------|-------------|-----|------|
| (없음) | | | | | | |

---

## 미해결 결정 / 블로커
- **I-04 (DiffusionDet 재현치 + P0a 진단 미수행)** 만 open. **I-05 (jq 미설치)** 는 blocked (Dockerfile 갱신 완료, rebuild 대기). 상세는 [ISSUE.md](./ISSUE.md).
- ~~I-01 (pyproject.toml)~~ → **R-05 resolved**. ~~I-03 (.gitignore ai-ml)~~ → **R-06 resolved**.
- ~~I-02 (env_docker/ 미생성)~~ → **R-03 resolved** (`/docker-init` 으로 해소).
- I-04 의 두 부분: (a) DiffusionDet 재현치 매칭(P0), (b) **메커니즘 진단 5행 채우기(P0a)**. 둘 다 통과해야 P1 FM 전환 시작.
- 사용자가 W&B 첫 도입 — sweep 도입 시점은 미정 (Hydra `--multirun` 으로 시작, 익숙해진 후 검토). [WANDB_GUIDE.md](./WANDB_GUIDE.md) "Sweep" 섹션.

## 컨텍스트 회복용 빠른 링크
- 헌법: [../CLAUDE.md](../CLAUDE.md) (CRITICAL 4개 + 마일스톤 + 이슈/상태 관리)
- 무엇을·왜: [PRD.md](./PRD.md) — 한 줄 목적 / 성공 기준 (P0/P0a/P1~/문서 산출물)
- 어떻게: [ARCHITECTURE.md](./ARCHITECTURE.md) — 루트 평탄 + Hydra + env_docker 생성됨
- 결정: [ADR.md](./ADR.md) — 5개 ADR
- ablation·진단 운영: [EXPERIMENTS.md](./EXPERIMENTS.md) — 단계적 로드맵 (P0/P0a/P1~P4) + 진단 카탈로그 + **실험 1회의 완료 정의 5단계**
- 모델 카드: [MODEL_CARD.md](./MODEL_CARD.md) — 변종 카탈로그 + **메커니즘 진단 결과 표**
- 이슈 트래커: [ISSUE.md](./ISSUE.md)
- Hydra 학습: [HYDRA_GUIDE.md](./HYDRA_GUIDE.md)
- W&B 학습: [WANDB_GUIDE.md](./WANDB_GUIDE.md)
- 레퍼런스 구현 (읽기 전용): [../DiffusionDet/README.md](../DiffusionDet/README.md)
- **환경 구성**: [../Makefile](../Makefile) · [../env_docker/Dockerfile](../env_docker/Dockerfile) · [../env_docker/docker-compose.yml](../env_docker/docker-compose.yml) · [../.env.example](../.env.example) · [../requirements.txt](../requirements.txt)
