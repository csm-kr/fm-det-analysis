# Pensieve — fm-det 의 지금 상태 한 페이지

> **이 문서가 답하는 질문**: 오랜만에 돌아왔을 때 *지금 어디까지 왔고, 다음에 무엇을 해야 하는가*? 본 문서는 **시점 의존 스냅샷** — 다른 문서들(PRD/ARCH/ADR/EXPERIMENTS/ISSUE) 가 *누적·정책* 인 반면, pensieve 는 *덮어쓰는 한 페이지*. 한 화면에서 컨텍스트가 회복되도록 짧게 유지.

> **갱신 규칙**: 매 작업 종료 시 본 문서를 갱신한다. **"마지막 업데이트" 한 줄·"지금 어디"·"다음 한 가지" 세 곳은 반드시 최신**. 나머지는 변경된 경우에만.

---

## 마지막 업데이트
- **일시**: 2026-05-22
- **갱신자**: Claude (호스트 rebuild 검증 → R-07 (jq/unzip) / R-08 (sm_120) 둘 다 resolved. I-08 (execute.py `_verify_success_metric` 의 `{run_dir}` 없는 code-only success_metric 강제 error 버그) patch + resolved. I-07 신규 — torch-cache named volume 영구화 깨짐, TORCH_HOME=/workspace/fm-det/.cache/torch workaround. model-diffusiondet code-only step 0/1/2/4 + loss-diffusiondet code-only step 0/1/3 status pending → completed.)

---

## 지금 어디 (현재 단계)
- **전체 단계**: 컨테이너 rebuild 검증 통과 (`torch 2.7.1+cu128`, `arch_list` 에 sm_120 포함, Blackwell forward PASS). `code-skeleton-loaders` 4/4 완료 ✅. model-diffusiondet code-only step (0/1/2/4) + loss-diffusiondet code-only step (0/1/3) 모두 completed. **남은 step**: model step 3 model-sanity (CP-2), loss step 2 loss-sanity, entrypoints-evals 4 step 전부 — 모두 step.md 미작성 상태.
- **활성 phase**: `model-diffusiondet` (4/5 done, step 3 model-sanity-overfit pending CP-2) · `loss-diffusiondet` (3/4 done, step 2 loss-sanity-50step pending) · `entrypoints-evals` (0/4 pending, step.md 전무).
- **활성 작업**: 남은 sanity step 2개 + entrypoints-evals 4 step 진행 — 각 phase 의 step.md 작성 후 execute.py.

## 다음 한 가지 (Single Next Action)
> 막연한 "이것저것" 대신 **다음에 손댈 한 가지**를 적는다. 끝나면 다음 한 가지로 갱신.

**`phases/model-diffusiondet/step3.md` 작성 (model-sanity-overfit, CP-2). 내용: 1 batch 100-step overfit + forward shape + all-params-have-grad + param_count_m + `runs/model-sanity-{ts}/sanity.json` 산출. 그 다음 `python3 scripts/execute.py model-diffusiondet` → CP-2 에서 stop 후 사용자 검토. (TORCH_HOME 설정 inherit — I-07 workaround.) 동일 패턴으로 loss-diffusiondet step2.md (loss-sanity-50step) 작성. 마지막으로 entrypoints-evals 의 4 step.md 모두 작성 후 execute.py 로 evals/coco.py + evals/voc.py + train/eval/infer.py + dry-run-1iter.**

이후 순서 (참고만):
1. ~~M0 부트스트래핑~~ ✅
2. ~~M1 data-sanity-coco val~~ ✅
3. ~~M2 데이터 sanity 전체 + Hydra base + datasets 구현~~ ✅
4. ~~컨테이너 rebuild (R-07/R-08)~~ ✅ — torch 2.7.1+cu128 + sm_120 forward + jq/unzip PASS.
5. ~~`code-skeleton-loaders` phase 4/4~~ ✅ (execute.py).
6. ~~`model-diffusiondet` code 4/5 (backbone/sampler/decoder/readme)~~ ✅ — step 3 model-sanity 남음.
7. ~~`loss-diffusiondet` code 3/4 (matcher/criterion/readme)~~ ✅ — step 2 loss-sanity 남음.
8. **(지금)** **`model-diffusiondet` step3.md 작성 → execute.py → CP-2 사용자 검토.**
9. **`loss-diffusiondet` step2.md 작성 → execute.py.**
10. **`entrypoints-evals` 4 step.md 작성 → execute.py** (evals/{coco,voc}.py + train/eval/infer.py + dry-run-1iter, GPU 필요).
11. **P0** `coco-repro-baseline` — 학습 (61 epoch, ~며칠). COCO val AP 46.2 ± 0.5 매칭 (I-04).
12. **P0 VOC** `voc-repro-baseline` — VOC07 test mAP@0.5 자체 baseline.
13. 미달 시 3-seed × 향상 요소 ablation.
14. **P0a 메커니즘 진단 5행** — `coco-diag-signal-scale / box-renewal / iter-step / num-boxes / nms-iou`.
15. P0a 5행 OK → P1 `coco-fm1-sampler-cfm`.

---

## 최근 변경 (최근 5개, 시간 역순)
- **2026-05-22** — **rebuild 검증 (R-07/R-08 resolved) + I-08 patch + I-07 신규 + model/loss code-only status 동기화**: 호스트 rebuild 후 `torch 2.7.1+cu128`, `arch_list=[..., sm_120, compute_120]`, RTX PRO 6000 Blackwell forward PASS, jq-1.6 / unzip 6.00 모두 동작 → R-07 (jq/unzip) / R-08 (sm_120) resolved. execute.py 의 `_verify_success_metric` 버그 (`{run_dir}` 없는 code-only step 도 run_dirs 강제) patch → I-08 resolved. I-07 신규 — torch-cache named volume 영구화 깨짐 + root 소유 / TORCH_HOME=/workspace/fm-det/.cache/torch workaround + ~/.bashrc 영구화 + .gitignore .cache/ 추가. ResNet50 가중치 97.8MB 재다운. model-diffusiondet step 0/1/2/4 + loss-diffusiondet step 0/1/3 status pending → completed (success_metric 직접 PASS 검증 후 summary 기록).
- **2026-05-22** — **`code-skeleton-loaders` phase 4 step 전부 완료**: execute.py 로 진행. step 0 transforms-common / step 1 coco-dataset-loader / step 2 voc-dataset-loader / step 3 hydra-configs 모두 completed + `runs/code-skeleton-loaders-{coco,voc}-...` 산출. hydra.compose(train, data=coco|voc) 합성 PASS, batch_size=16 한 자리. (I-08 patch 가 step 0 의 `test -f ... && python3 -c '...'` success_metric 을 정상 통과시키는 결정타.)
- **2026-05-22** — **`code-skeleton-loaders` step 2 `voc-dataset-loader` 재검증 + index.json 정리**: 이전 시도에 stale `crash_reason: Unknown` 마커로 status=pending 남아 있었으나 코드는 기존 작성분 그대로 동작. AC 재실행: voc07-trainval batch-size=2 seed=42 → sanity_pass=true, batch_shape=[2,3,800,1088], num_targets_per_image=[1,1], cat_idx_range=[3,18]. jq AC 통과. index.json crash 마커 제거 + status=completed + summary 갱신.
- **2026-05-22** — **`code-skeleton-loaders` step 2 `voc-dataset-loader` 완료**: `datasets/voc/sanity_loader.py` 신설 — OmegaConf 로 configs/data/voc.yaml 로드 + named split (voc07-trainval/voc07-test/voc12-trainval/voc-trainval-combined) → cfg.train_split / eval_split 오버라이드 후 build_voc_loader 호출. trainval 계열은 train 모드 (drop_difficult=True). AC PASS: batch_shape=[2,3,800,1088], num_targets_per_image=[1,1], cat_idx_range=[3,18], split=voc07-trainval, sanity_pass=true. jq AC 통과.
- **2026-05-21** — **`code-skeleton-loaders` step 1 `coco-dataset-loader` 완료**: `datasets/coco/sanity_loader.py` 신설 — OmegaConf 로 configs/data/coco.yaml 로드 + batch_size override + `build_coco_loader(split='eval')` 1-batch 검증. sanity.json 산출 (batch_shape=[2,3,800,1248], num_targets_per_image=[19,14], cat_idx_range=[0,72], sanity_pass=true). jq AC 통과.

## 진행 중 phase

| phase tag | 상태 | step 진행 | runs/ |
|-----------|------|----------|-------|
| `data-sanity-coco` | completed (CP-1 ✅) | 0,1,2,3 ✅ | `data-sanity-{download,analyze,vis,report}-20260521-{1332,1335}` |
| `data-sanity-coco-train` | completed (CP-1 auto-approved) | 0,1,2,3 ✅ | `data-sanity-{download-1413, analyze-1509, vis-1510, report-1510}` |
| `data-sanity-voc` | completed (CP-1 ✅) | 0,1,2,3 ✅ | `data-sanity-voc-{download-1429, analyze-1448, vis-1448, report-1448}` |
| `code-skeleton-loaders` | completed (0,1,2,3 ✅) | transforms-common ✅ / coco-dataset-loader ✅ / voc-dataset-loader ✅ / hydra-configs ✅ | `code-skeleton-loaders-coco-20260521-{222054,222242}`, `code-skeleton-loaders-voc-20260521-222615` |
| `model-diffusiondet` | 4/5 done — step 3 model-sanity (CP-2) pending | 0 backbone ✅ / 1 sampler ✅ / 2 decoder ✅ / 3 model-sanity ⏸ / 4 readme-mermaid ✅ | — |
| `loss-diffusiondet` | 3/4 done — step 2 loss-sanity pending | 0 matcher ✅ / 1 criterion ✅ / 2 loss-sanity ⏸ / 3 readme-mermaid ✅ | — |
| `entrypoints-evals` | **0/4 pending (step.md 전무)** | — | — |
| `coco-repro-baseline` | pending (P0) | — | — |
| `voc-repro-baseline` | pending | — | — |

## 최근 ablation / 진단 결과 (최근 3개)
없음 — P0/P0a 미시작. EXPERIMENTS.md 의 진단 표 + ablation 표가 채워지기 시작하면 본 섹션에 최근 3개 행 미러링.

| run_id | dataset | phase | 변경 1변수 | AP / mAP@.5 | ΔAP | 결론 |
|--------|---------|-------|-----------|-------------|-----|------|
| (없음) | | | | | | |

---

## 미해결 결정 / 블로커
- **I-04 (DiffusionDet 재현치 + P0a 진단 미수행)** open. **I-07 (torch-cache named volume 영구화 깨짐 + mount-point root 소유)** blocked — TORCH_HOME=/workspace/fm-det/.cache/torch workaround 적용 중, Dockerfile pre-create+chown patch 가 영구 fix.
- ~~I-05 (jq/unzip)~~ → **R-07 resolved** (rebuild). ~~I-06 (PyTorch sm_120 Blackwell)~~ → **R-08 resolved** (torch 2.7.1+cu128 rebuild). ~~I-08 (execute.py `_verify_success_metric` `{run_dir}` 버그)~~ → resolved (scripts/execute.py patch).
- ~~I-01 (pyproject.toml)~~ → **R-05 resolved**. ~~I-03 (.gitignore ai-ml)~~ → **R-06 resolved**. ~~I-02 (env_docker/)~~ → **R-03 resolved**.
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
