# Pensieve — fm-det 의 지금 상태 한 페이지

> **이 문서가 답하는 질문**: 오랜만에 돌아왔을 때 *지금 어디까지 왔고, 다음에 무엇을 해야 하는가*? 본 문서는 **시점 의존 스냅샷** — 다른 문서들(PRD/ARCH/ADR/EXPERIMENTS/ISSUE) 가 *누적·정책* 인 반면, pensieve 는 *덮어쓰는 한 페이지*. 한 화면에서 컨텍스트가 회복되도록 짧게 유지.

> **갱신 규칙**: 매 작업 종료 시 본 문서를 갱신한다. **"마지막 업데이트" 한 줄·"지금 어디"·"다음 한 가지" 세 곳은 반드시 최신**. 나머지는 변경된 경우에만.

---

## 마지막 업데이트
- **일시**: 2026-05-21
- **갱신자**: Claude (`code-skeleton-loaders` step 1 `coco-dataset-loader` 완료 — `datasets/coco/sanity_loader.py` 신설 + 1-batch sanity PASS: batch_shape=[2,3,800,1248], num_targets=[19,14], cat_idx_range=[0,72]. jq AC 통과.)

---

## 지금 어디 (현재 단계)
- **전체 단계**: 그룹 B 의 datasets / models / losses 코드 작성 완료 + CPU sanity 통과. **I-06 (Blackwell sm_120 PyTorch 호환) 블로커** — Dockerfile patch 완료, 호스트 rebuild 대기. rebuild 후 evals/ → train.py/eval.py/infer.py → P0 학습 흐름.
- **활성 phase**: `code-skeleton-loaders` (step 0 ✅ + step 1 `coco-dataset-loader` ✅ — step 2 voc sanity / step 3 hydra-configs 남음) / `model-diffusiondet` (코드 + README mermaid 완료, model-sanity GPU 50-step 은 rebuild 후) / `loss-diffusiondet` (코드 + README mermaid 완료, loss-sanity GPU 50-step rebuild 후).
- **활성 작업**: 호스트에서 컨테이너 rebuild 대기.

## 다음 한 가지 (Single Next Action)
> 막연한 "이것저것" 대신 **다음에 손댈 한 가지**를 적는다. 끝나면 다음 한 가지로 갱신.

**`code-skeleton-loaders` step 2 `voc-dataset-loader` 진행 — `datasets/voc/sanity_loader.py` 신설 + 1-batch 검증 (split=voc07-test, batch-size 2). 이후 step 3 hydra-configs (configs/data/{coco,voc}.yaml OmegaConf 로드 검증). 병행: 호스트 `make build && make up && make nvidia-test` 로 I-06 해소 (Dockerfile base `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` 2차 patch 적용 후 sm_120 검증) — GPU 학습/평가 흐름 unblocked.**

이후 순서 (참고만):
1. ~~M0 부트스트래핑~~ ✅
2. ~~M1 data-sanity-coco val~~ ✅
3. ~~M2 데이터 sanity 전체 + Hydra base + datasets 구현~~ ✅
4. ~~`model-diffusiondet` 코드 + README mermaid (110.7M params)~~ ✅ (CPU sanity / GPU sanity rebuild 후)
5. ~~`loss-diffusiondet` 코드 + README mermaid~~ ✅ (loss 38.5 finite, 314/314 grad CPU)
6. **(지금)** **호스트 `make build && make up && make nvidia-test`** — Dockerfile patch (PyTorch 2.6.0 + jq/unzip + torch-cache volume) 적용. I-05 / I-06 동시 해소.
7. GPU 검증 후 `entrypoints-evals` — `evals/{coco,voc}.py` + `train.py` / `eval.py` / `infer.py` Hydra 진입점.
8. model-sanity / loss-sanity GPU 50-step.
9. **P0** `coco-repro-baseline` — 학습 (61 epoch, ~며칠). COCO val AP 46.2 ± 0.5 매칭 (I-04).
10. **P0 VOC** `voc-repro-baseline` — VOC07 test mAP@0.5 자체 baseline.
11. 미달 시 3-seed × 향상 요소 ablation (runs/report 정리).
12. **P0a 메커니즘 진단 5행** — `coco-diag-signal-scale / box-renewal / iter-step / num-boxes / nms-iou`.
13. P0a 5행 OK → P1 `coco-fm1-sampler-cfm`.

---

## 최근 변경 (최근 5개, 시간 역순)
- **2026-05-21** — **`code-skeleton-loaders` step 1 `coco-dataset-loader` 완료**: `datasets/coco/sanity_loader.py` 신설 — OmegaConf 로 configs/data/coco.yaml 로드 + batch_size override + `build_coco_loader(split='eval')` 1-batch 검증. sanity.json 산출 (batch_shape=[2,3,800,1248], num_targets_per_image=[19,14], cat_idx_range=[0,72], sanity_pass=true). jq AC 통과. step 0 시각으로 status 반영.
- **2026-05-21** — **`code-skeleton-loaders` step 0 status 반영**: `datasets/transforms.py` (`build_transforms` + Compose/RandomResize/RandomHorizontalFlip/ToTensor/Normalize + collate_fn) 는 이전 세션에 작성 완료 + AC PASS 상태였음. phases/code-skeleton-loaders/index.json step 0 status: pending → completed + summary 한 줄 기록.
- **2026-05-21** — **Dockerfile 2차 patch — PyTorch 2.7.1+cu128 (sm_120 공식 지원)**: 1차 patch (2.6.0+cu124) 가 rebuild 후 `torch.cuda.get_arch_list()` 에 sm_120 미포함으로 no-kernel-image 재현. 웹 검증 결과 **PyTorch sm_120 첫 공식 stable = 2.7.0** (cu128 wheel). `env_docker/Dockerfile` base 를 `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel` 로 갱신 + requirements.txt 주석 동기화. ISSUE.md I-06 에 1차/2차 patch 이력 + arch_list 검증 컨벤션 추가. 호스트 driver CUDA 13 forward-compat 확인. 호스트 `make build && make up` 재실행 대기.
- **2026-05-21** — **models/ + losses/ 구현 + Dockerfile PyTorch 2.6.0 patch (rebuild 대기)**: `models/{backbone,sampler,decoder,diffusiondet}.py` (110.7M params, eval shape `[B,N,C/4]` / train shape `[B,K=6,N,C/4]`) + `models/README.md` mermaid 2개 (전체 구조 + DetectionHead 내부) + 사전학습 가중치 위치 안내 (`/home/docker_user/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth` 97.8MB, `torch-cache` named volume 영구화). `losses/{matcher,criterion}.py` (SimOTA dynamic_k + focal/L1/GIoU + deep supervision K=6) + `losses/README.md` mermaid 2개 — CPU sanity loss=38.5 finite, 314/314 trainable params 에 grad. **I-06 신규**: Blackwell sm_120 + PyTorch 2.5.1 미지원 → Dockerfile base patch (`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`) + requirements 주석 갱신 — 호스트 `make build && make up && make nvidia-test` 대기. 동일 rebuild 가 I-05 (jq/unzip apt) + torch-cache volume 도 함께 적용.
- **2026-05-21** — **M2 도달 — 그룹 B 코드 시작**: `data-sanity-coco-train` 완료 (118,287 / 860,001 ann), `data-sanity-voc` 완료 (CP-1 approved). `configs/` Hydra base 7 yaml — DiffusionDet 동치 (lr=2.5e-5 / epochs=61 / batch=16). `datasets/transforms.py` (short 800-1333 + flip 0.5) + `datasets/coco/dataset.py` + `datasets/voc/dataset.py` PyTorch Dataset 구현. 1-batch loading sanity PASS.

## 진행 중 phase

| phase tag | 상태 | step 진행 | runs/ |
|-----------|------|----------|-------|
| `data-sanity-coco` | completed (CP-1 ✅) | 0,1,2,3 ✅ | `data-sanity-{download,analyze,vis,report}-20260521-{1332,1335}` |
| `data-sanity-coco-train` | completed (CP-1 auto-approved) | 0,1,2,3 ✅ | `data-sanity-{download-1413, analyze-1509, vis-1510, report-1510}` |
| `data-sanity-voc` | completed (CP-1 ✅) | 0,1,2,3 ✅ | `data-sanity-voc-{download-1429, analyze-1448, vis-1448, report-1448}` |
| `code-skeleton-loaders` | step 0 ✅ (1/2/3 pending) | step 0 transforms-common ✅ — datasets/{coco/dataset.py, voc/dataset.py} + configs/data/{coco,voc}.yaml 코드는 있으나 step status 미반영 | — |
| `model-diffusiondet` | 코드 + README mermaid 완료 / GPU model-sanity rebuild 후 | models/{backbone, sampler, decoder, diffusiondet}.py + models/README.md + utils/box_ops.py + configs/model/diffusiondet.yaml (110.7M params, CPU forward+backward ✅) | — |
| `loss-diffusiondet` | 코드 + README mermaid 완료 / GPU loss-sanity rebuild 후 | losses/{matcher, criterion}.py + losses/README.md + configs/loss/diffusion.yaml (CPU loss=38.5 finite, 314/314 grad) | — |
| `entrypoints-evals` | **pending (rebuild 후)** | — | — |
| `coco-repro-baseline` | pending (P0) | — | — |
| `voc-repro-baseline` | pending | — | — |

## 최근 ablation / 진단 결과 (최근 3개)
없음 — P0/P0a 미시작. EXPERIMENTS.md 의 진단 표 + ablation 표가 채워지기 시작하면 본 섹션에 최근 3개 행 미러링.

| run_id | dataset | phase | 변경 1변수 | AP / mAP@.5 | ΔAP | 결론 |
|--------|---------|-------|-----------|-------------|-----|------|
| (없음) | | | | | | |

---

## 미해결 결정 / 블로커
- **I-04 (DiffusionDet 재현치 + P0a 진단 미수행)** 만 open. **I-05 (jq 미설치)** blocked. **I-06 (PyTorch sm_120 미지원 — Blackwell)** 신규 blocked — `pip install --pre torch ... cu126` 또는 Dockerfile rebuild 필요. 상세는 [ISSUE.md](./ISSUE.md).
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
