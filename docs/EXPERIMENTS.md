# 실험 운영

> **이 문서의 위상**: ablation 중심 프로젝트(fm-det)의 *중심 산출물*. 모든 실험은 이 표에 한 행씩 누적되며, 표를 떠난 결과는 "안 한 것" 으로 간주.

## 디렉터리 컨벤션
모든 학습 실행은 다음 구조의 결과를 남긴다:

```
runs/{YYYYMMDD-HHmm}-{tag}/
├── config.yaml         # 실행 시점의 하이퍼파라미터 스냅샷 (자동 복사)
├── git_rev.txt         # 코드 커밋 해시
├── seed.txt            # 시드값
├── data_manifest.json  # 데이터셋 SHA256 + 샘플 수
├── metrics.csv         # step/epoch별 메트릭 시계열
├── checkpoints/
│   ├── best.pt         # val AP 최고
│   └── last.pt
├── tb_logs/            # TensorBoard event
├── wandb/              # W&B 로컬 디렉터리 (offline mode 우선)
└── eval.json           # 학습 종료 후 scripts/eval.py 산출 — MODEL_CARD 가 인용
```

### tag 명명 규칙
`{dataset}-{phase}-{module}-{변경 1변수}` 형식. `dataset` ∈ `{coco, voc}` — 데이터셋은 **각각 독립 학습** 이므로 모든 tag 의 prefix 가 된다 ([DATA_CARD.md](./DATA_CARD.md)). 예:
- `coco-repro-baseline` (COCO 재구현 단계, 변경 없음)
- `voc-repro-baseline` (VOC 재구현 단계 — COCO 와 별도 학습)
- `coco-fm1-sampler-cfm` (COCO 에서 FM 1차 전환, sampler 를 conditional FM 으로)
- `coco-abl-encoder-swin` (COCO ablation, encoder 를 swin 으로)
- `voc-abl-head-shallow6` (VOC ablation, head 깊이를 6 으로)

## 시드 정책 (CRITICAL — CLAUDE.md)
- 모든 학습/평가 스크립트는 `--seed` 인자를 **필수**로 받는다 (argparse `required=True`). 없으면 시작 차단.
- 시드 고정 범위: `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`.
- DataLoader 의 `worker_init_fn` 도 시드 고정.
- **default 시드 = 42** (단일 실행). 결론 후보 변종은 **시드 3개 (42, 17, 2024) 평균 ± 표준편차** 로 재실행해 보고.
  - GPU 부담을 고려해 처음 ablation 1회는 단일 시드, 의미 있는 ΔAP 가 보일 때만 3-시드 재평가. EVAL_PROTOCOL.md 의 통계적 유의성 절차 참고.

## 변수 격리 규칙 (CRITICAL — CLAUDE.md)
ablation 의 한 행은 baseline 대비 **변경한 1 변수** 만 다르다. 변경한 변수는 ablation 표 컬럼에 명시.
- 변경 가능한 변수 카테고리: (a) 모듈 (encoder/decoder/head/sampler/loss/matcher), (b) 하이퍼파라미터 (lr/wd/batch/iter/scheduler), (c) 데이터 (augmentation 강도, 데이터셋).
- 한 행에서 2개 이상 변수가 동시에 바뀌면 그 행의 ΔAP 는 표에서 제외 (또는 "혼합" 마크 + 결론 보류).

## 하이퍼파라미터 스냅샷
- 학습 시작 직후 `configs/{name}.yaml` 을 `runs/{id}/config.yaml` 로 그대로 복사.
- 같은 시점에 `git rev-parse HEAD` 결과를 `git_rev.txt` 에 기록.
- `data_manifest.json` 에 `coco/annotations/*.json` 과 `voc/{2007,2012}/ImageSets/*.txt` 의 SHA256 + 샘플 수 기록.
- 코드·설정·데이터 셋 변경 추적 가능 — "이 결과를 어떻게 만들었는가" 답변 의무.

## Config 시스템 (Hydra)
fm-det 은 **Hydra 1.3** 으로 config 를 관리한다. 자세한 개념·사용법·함정은 [HYDRA_GUIDE.md](./HYDRA_GUIDE.md). 본 문서는 ablation 운영 관점의 컨벤션만.

- **ablation 한 행 = `configs/experiment/{tag}.yaml` 한 파일 = `runs/{ts}-{tag}/` 한 디렉터리** 의 1:1 매핑.
- `configs/experiment/{tag}.yaml` 안에서 `defaults:` 로 base group (`model/`, `data/`, `train/`, `loss/`) 을 선택하고, 그 아래 한 변수만 override 한다. 변경 변수가 즉시 가시화됨 → CRITICAL "단일 변수 ablation" 의 자연스러운 강제.
- 실행: `python train.py +experiment={tag} seed=42`
- Hydra 의 `hydra.run.dir` 을 `runs/{now:%Y%m%d-%H%M}-${experiment.tag}/` 로 override 해 기존 `runs/{ts}-{tag}/` 컨벤션과 자동 일치.

## 실험 추적 도구
- **TensorBoard (기준)**: 모든 `runs/{id}/tb_logs/` 자동 기록. 외부 의존 0, 오프라인.
- **W&B (병행, 학습 동반)**: `runs/{id}/wandb/` 에 로컬 기록. 자세한 사용법·튜토리얼·함정은 [WANDB_GUIDE.md](./WANDB_GUIDE.md).
  - 환경변수: `WANDB_API_KEY` (컨테이너 entrypoint 에서 주입, repo 에 절대 커밋 금지).
  - 프로젝트 이름: `fm-det`.
- 기록 항목: `train/loss`, `val/loss`, `val/AP`, `val/AP50`, `val/AP75`, `lr`, `epoch`, `GPU memory`, `samples/sec`.

### 도구 학습 노트 누적 (시간순)
> 첫 도입 도구 (Hydra · W&B) 의 사용 중 막힌 부분·알게 된 기능을 한 줄씩 누적. 새 phase 마다 추가.
- `{YYYY-MM-DD}` — `[hydra|wandb]` — `{phase tag}` — `{배운 점 한 줄}`

## 데이터 버전
- 데이터셋 변경 시 `data/CHANGELOG.md` 에 한 줄 의무.
- 형식: `YYYY-MM-DD - {변경 한 줄} - {새 샘플 수}`.
- 학습 스크립트는 `data_manifest.json` 으로 자동 검증 — 이전 manifest 와 SHA256 불일치 시 경고 + 사용자 확인 후 진행.

---

## 단계적 FM 전환 로드맵 (전체 phase 지도)

각 phase 는 **데이터셋 별로 독립 실행** — COCO 와 VOC 트랙이 별도 진행된다. tag prefix 가 `coco-` 와 `voc-` 로 갈린다.

| Phase | tag prefix | 목표 | 비교 baseline | 변경 변수 |
|-------|-----------|------|--------------|----------|
| **Pre-P0 데이터 sanity** | `data-sanity-{ds}` | 다운로드 무결성 + 분포·박스·해상도 통계 + 시각화 4종 + report.md (CP-1 사용자 검토). 학습 시작 전 데이터 정합 확인. | (없음) | (없음 — 데이터셋만 차원) |
| **P0 Repro (COCO)** | `coco-repro-*` | `DiffusionDet/` 동치 재구현 → COCO AP 46.2 ± 0.5 | DiffusionDet 본 repo (AP 46.2) | (없음) |
| **P0 Repro (VOC)** | `voc-repro-*` | VOC 학습 baseline 확립 (자체 기준) | (없음 — 본 프로젝트 자체 baseline) | (없음) |
| **P0a 메커니즘 진단** | `{ds}-diag-*` | **DiffusionDet 의 핵심 메커니즘을 실험적으로 측정** — 재구현 코드 위에서 1 변수씩 변경해 영향 정량화. 단순한 ablation 이 아니라 *DiffusionDet 의 동작을 이해하기 위한* 실험. | 같은 데이터셋의 P0 best | signal scale / box renewal on/off / iter step / num eval boxes / NMS 임계 등 |
| **P1 FM-1차** | `{ds}-fm1-*` | sampler + loss 만 flow matching (ODE/vector field) | 같은 데이터셋의 P0 best | sampler+loss (1쌍) |
| **P2 FM-2차** | `{ds}-fm2-*` | matcher / head 도 FM 속성에 맞게 재설계 | 같은 데이터셋의 P1 best | matcher 또는 head |
| **P3 모듈 abl.** | `{ds}-abl-*` | encoder/decoder/head 각 ablation | 같은 데이터셋의 P1/P2 best | 모듈 1개 |
| **P4 ablation 종합** | `{ds}-final-*` | 최선 변종 3-시드 재실행 + 보고 | 같은 데이터셋의 P0/P1/P2 best | (재실행) |

(`{ds}` ∈ `{coco, voc}`)

### P0a 메커니즘 진단 실험 카탈로그 (DiffusionDet 의 동작을 실증)
> 본 카탈로그는 fm-det 의 **내공 형성 본질** — DiffusionDet 을 "코드만 따라 적었다" 가 아니라 "주요 노브를 직접 돌려보고 효과를 표로 안다" 까지를 목표. P1 FM 전환 시작 전 최소 5개 행이 채워져야 한다.

| 진단 tag | 변경 변수 | 측정 가설 (DiffusionDet 의 어떤 동작을 알게 되는가) |
|---------|----------|-------------------------------------|
| `{ds}-diag-signal-scale` | signal scale ∈ {1.0, 2.0(base), 4.0} | 박스 좌표 노이즈의 신호 강도가 학습 안정성·AP 에 미치는 영향 — DiffusionDet 의 noise schedule 직관. |
| `{ds}-diag-box-renewal` | box renewal on / off | 본 repo "wo box renewal" 변종 (AP 45.7) 의 -0.5 AP 손해를 실측 재현 + 어느 step 에서 가장 큰 영향인지 분석. |
| `{ds}-diag-iter-step` | iter step ∈ {1, 2, 4(base), 8} | denoising step 수와 AP 의 trade-off — FM 전환 후 같은 곡선을 그려 비교 baseline 이 됨. |
| `{ds}-diag-num-boxes` | num eval boxes ∈ {100, 300, 500(base), 1000} | proposal 수가 recall·AP·FPS 에 미치는 영향. |
| `{ds}-diag-nms-iou` | NMS IoU 임계 ∈ {0.3, 0.5(base), 0.7} | 후처리가 AP 에 미치는 영향 — 모델 변경 효과와 후처리 효과를 분리. |
| `{ds}-diag-noise-vis` | (관찰만) 학습 step별 박스 좌표 분포 시각화 | 노이즈 분포의 진화 — TB image + W&B 에 산점도. AP 측정 없이 정성적 이해. |

진단 실험의 결론은 [MODEL_CARD.md#메커니즘-진단-결과](./MODEL_CARD.md) 의 진단 결과 표에 미러링.

각 phase 의 step 파일은 `/harness` 가 `phases/{task}/step{N}.md` 로 생성. 본 표는 phase 진행 중 갱신.

---

## 실험 1회의 완료 정의 (workflow)
> **CRITICAL ([CLAUDE.md#개발-프로세스](../CLAUDE.md))**: 학습만 돌리고 표/카드/ADR 미갱신은 "실험 안 한 것". 한 run 의 완료는 다음 5단계가 모두 끝났을 때.

1. **학습 실행** — `python train.py +experiment={tag} seed=42` → `runs/{ts}-{tag}/` 자동 생성.
2. **평가 산출** — `python eval.py +experiment={tag} run_dir=runs/{ts}-{tag}` → `runs/{ts}-{tag}/eval.json`.
3. **표 갱신** — 본 문서 ablation 표 또는 진단 표에 한 행 append (run_id, dataset, 변경 1변수, seed, AP/mAP, ΔAP, FPS).
4. **결론 한 줄 작성** — *DiffusionDet 또는 FM 의 어떤 동작을 알게 되었는가* 를 한 줄. "AP 가 0.3 떨어짐" 이 아니라 "iter step 1 에서 작은 객체(APs) 가 가장 큰 손해 → denoising 횟수가 small object recall 의 병목" 같은 *해석*.
5. **카드/ADR 미러링** — 결론이 *결정 가치* 가 있으면 [ADR.md](./ADR.md) 한 행 신설. *모델 변종 가치* 가 있으면 [MODEL_CARD.md](./MODEL_CARD.md) 의 변종 카탈로그 또는 진단 결과 표에 미러링.

5단계 중 하나라도 빠지면 그 run 의 `runs/{tag}/.status` 에 `INCOMPLETE` 표기 + 다음 작업 시작 전 마무리.

---

## Ablation 표 (누적 — 이 프로젝트의 핵심 산출물)

> 본 표가 fm-det 의 결론. 각 행은 변경한 1 변수와 측정된 ΔAP 를 가진다. 의미 있는 변종은 3-시드 재평가.

| run_id (`runs/...`) | dataset | phase | 변경 1변수 | seed | AP (COCO) / mAP@.5 (VOC) | ΔAP vs base | FPS | 결론 한 줄 |
|---------------------|---------|-------|-----------|------|-------------------------|------------|-----|-----------|
| _(채워가는 표 — 학습 종료 시 자동 한 줄 append + 본인 검수 후 결론 컬럼 채움. dataset 컬럼이 coco 인 행은 COCO AP, voc 인 행은 VOC mAP@.5)_ | | | | | | | | |

기록 자동화: `scripts/eval.py` 가 학습 종료 시 `runs/{id}/eval.json` 을 생성하고, `scripts/append_experiments.py` (별도 phase 에서 도입) 가 본 표에 한 줄 추가.

## 관련 문서
- 결정 근거: [ADR.md](./ADR.md)
- 평가 프로토콜 (변경 금지): [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
- 모델 변종 카드: [MODEL_CARD.md](./MODEL_CARD.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
