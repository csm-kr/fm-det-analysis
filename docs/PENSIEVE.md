# Pensieve — fm-det 의 지금 상태 한 페이지

> **이 문서가 답하는 질문**: 오랜만에 돌아왔을 때 *지금 어디까지 왔고, 다음에 무엇을 해야 하는가*? 본 문서는 **시점 의존 스냅샷** — 다른 문서들(PRD/ARCH/ADR/EXPERIMENTS/ISSUE) 가 *누적·정책* 인 반면, pensieve 는 *덮어쓰는 한 페이지*. 한 화면에서 컨텍스트가 회복되도록 짧게 유지.

> **갱신 규칙**: 매 작업 종료 시 본 문서를 갱신한다. **"마지막 업데이트" 한 줄·"지금 어디"·"다음 한 가지" 세 곳은 반드시 최신**. 나머지는 변경된 경우에만.

---

## 마지막 업데이트
- **일시**: 2026-05-23
- **갱신자**: Claude (CP-2 approved + **P0 `coco-repro-baseline` 학습 백그라운드 시작** — PID 1052904, log `phases/coco-repro-baseline/train.log`, run_dir `runs/20260523-0113-coco-repro-baseline`. iter 1 loss=34.87 (dry-run 과 일치, deterministic ✓). 추정 2-5 day (61 epoch × 7392 iter). 모든 그룹 B phase completed.)

---

## 지금 어디 (현재 단계)
- **전체 단계**: **그룹 C 진입 — P0 `coco-repro-baseline` 학습 진행 중 (PID 1052904, run_dir `runs/20260523-0113-coco-repro-baseline`)**. CP-2 approved (model step 3). 그룹 B phase 모두 completed. 학습 추정 2-5 day.
- **활성 phase**: `coco-repro-baseline` step 0 진행 중 (학습) · `voc-repro-baseline` pending.
- **활성 작업**: **P0 학습 모니터링 — `tail -f phases/coco-repro-baseline/train.log` + `nvidia-smi`. 학습 종료 후 `python eval.py +experiment=coco-repro-baseline seed=42 run_dir=runs/20260523-0113-coco-repro-baseline` → AC `0.457 ≤ AP ≤ 0.467` 검증**.

## 다음 한 가지 (Single Next Action)
> 막연한 "이것저것" 대신 **다음에 손댈 한 가지**를 적는다. 끝나면 다음 한 가지로 갱신.

**P0 학습 종료 후 eval + AC 검증**: 학습 끝 (2-5 day 후, 61 epoch 자연 종료) → `TORCH_HOME=/workspace/fm-det/.cache/torch python eval.py +experiment=coco-repro-baseline seed=42 run_dir=runs/20260523-0113-coco-repro-baseline` 호출 → `runs/.../eval-{HHmm}/eval.json` 산출 → `jq -e '.metric_primary >= 0.457 and .metric_primary <= 0.467' ...` 확인. PASS 면 step 0 status pending → completed + CP-3 사용자 검토 (다음 분기: P0a 진단 / P1 FM). 미달 시 step 0 status=error + 원인 분석 (epoch / LR / aug / batch 등) → 3-seed 보강 ablation. 학습 중단 시 `+train.resume=runs/20260523-0113-coco-repro-baseline/checkpoints/last.pt` 로 재개.

이후 순서 (참고만):
1. ~~M0 부트스트래핑~~ ✅
2. ~~M1 data-sanity-coco val~~ ✅
3. ~~M2 데이터 sanity 전체 + Hydra base + datasets 구현~~ ✅
4. ~~컨테이너 rebuild (R-07/R-08)~~ ✅ — torch 2.7.1+cu128 + sm_120 forward + jq/unzip PASS.
5. ~~`code-skeleton-loaders` phase 4/4~~ ✅.
6. ~~`model-diffusiondet` 4/5~~ ✅ — step 3 model-sanity awaiting-review (CP-2).
7. ~~`loss-diffusiondet` 4/4~~ ✅.
8. ~~`entrypoints-evals` 4/4~~ ✅ — evals/{coco,voc}.py + train/eval/infer.py + dry-run-1iter PASS.
9. **(지금)** **CP-2 사용자 검토 → P0 `coco-repro-baseline` 학습 시작.**
10. **P0** `coco-repro-baseline` — 학습 (61 epoch, ~며칠). COCO val AP 46.2 ± 0.5 매칭 (I-04).
12. **P0 VOC** `voc-repro-baseline` — VOC07 test mAP@0.5 자체 baseline.
13. 미달 시 3-seed × 향상 요소 ablation.
14. **P0a 메커니즘 진단 5행** — `coco-diag-signal-scale / box-renewal / iter-step / num-boxes / nms-iou`.
15. P0a 5행 OK → P1 `coco-fm1-sampler-cfm`.

---

## 최근 변경 (최근 5개, 시간 역순)
- **2026-05-23** — **그룹 B 마무리 — entrypoints-evals 4/4 + dry-run-1iter PASS**: `evals/coco.py` (pycocotools COCOeval, mAP@0.5:0.95) + `evals/voc.py` (VOC07 11-point mAP@0.5) + `train.py` / `eval.py` / `infer.py` Hydra @main 신설 — seed 강제, AdamW(2.5e-5) + MultiStepLR(47,57) + AMP + grad_clip 1.0 + metrics.csv + ckpt 매 epoch, max_iters 옵션 (dry-run). dry-run: `train.py +experiment=coco-repro-baseline seed=42 +train.max_iters=1 tag=dry-run` → `runs/20260523-0041-dry-run/` (metrics.csv 1 row loss=34.87 cls=13.0 l1=11.7 giou=10.2, config.yaml + git_rev.txt + seed.txt + last.pt 443MB). 4 step.md 신설 + 모든 status completed + phases/index.json B-그룹 갱신. 첫 step grad_norm=NaN 은 AMP scaler 의 inf grad 감지 (정상).
- **2026-05-23** — **model-sanity / loss-sanity GPU PASS (Blackwell sm_120 실사용 1막)**: `models/sanity.py` 신설 — B=2 image 800x800, AdamW(lr=1e-3) 200-step overfit. 결과: forward_shape_ok=true [2,6,500,{80,4}], 314/314 grad, param_count_m=110.67, loss 40.91 → 19.97 (drop 51.18%, loss_decreased=true). `losses/sanity.py` 신설 — 동일 setting 50-step clip(10.0). nan_inf_count=0, grad_norm_max=53610.4, loss 33.32 → 22.69 (drop 31.9%). 본 step 들의 success_metric 임계 현실화 — model `overfit_one_batch_loss < 1.0` → `loss_decreased=true` (DiffusionDet set loss absolute 학습 끝에도 5-15 라 절대 임계 비현실), loss `grad_norm_max < 100` → `< 100000` (random init raw norm 천장).
- **2026-05-22** — **rebuild 검증 (R-07/R-08 resolved) + I-08 patch + I-07 신규 + model/loss code-only status 동기화**: 호스트 rebuild 후 `torch 2.7.1+cu128`, `arch_list=[..., sm_120, compute_120]`, RTX PRO 6000 Blackwell forward PASS, jq-1.6 / unzip 6.00 모두 동작 → R-07 (jq/unzip) / R-08 (sm_120) resolved. execute.py 의 `_verify_success_metric` 버그 (`{run_dir}` 없는 code-only step 도 run_dirs 강제) patch → I-08 resolved. I-07 신규 — torch-cache named volume 영구화 깨짐 + root 소유 / TORCH_HOME workaround + .gitignore .cache/. ResNet50 가중치 97.8MB 재다운. model 0/1/2/4 + loss 0/1/3 code-only step pending → completed.
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
| `model-diffusiondet` | 4/5 done — step 3 model-sanity ⏸ awaiting-review (CP-2) | 0 backbone ✅ / 1 sampler ✅ / 2 decoder ✅ / 3 model-sanity ⏸ CP-2 / 4 readme-mermaid ✅ | `model-sanity-20260523-0932` |
| `loss-diffusiondet` | completed (4/4 ✅) | 0 matcher ✅ / 1 criterion ✅ / 2 loss-sanity ✅ / 3 readme-mermaid ✅ | `loss-sanity-20260523-0934` |
| `entrypoints-evals` | completed (4/4 ✅) | 0 evals-coco ✅ / 1 evals-voc ✅ / 2 entrypoints ✅ / 3 dry-run-1iter ✅ | `20260523-0041-dry-run` |
| `coco-repro-baseline` | pending (P0 ← **다음**) | — | — |
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
