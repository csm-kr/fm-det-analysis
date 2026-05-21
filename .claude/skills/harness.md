이 프로젝트는 Harness 프레임워크의 **ai-ml 통합본** 으로 동작하는 `/harness` 를 사용한다. SWE 종류의 `/harness` 가 "코드 작성 → 빌드 → 테스트 통과 → 커밋" 사이클을 가정한다면, 이 정본은 그 자리에 박혀 ML 라이프사이클 9 단계 위에서 step 을 설계한다.

> **본 스킬도 [LLM_GUIDE.md](../../LLM_GUIDE.md) 의 4 원칙을 따른다.** 특히 *Think Before Coding* — 데이터·모델·loss 결정의 모호한 자리에서는 대안을 제시한다.
>
> **배포 경로**: 본 파일은 `.claude/skills/templates/ai-ml/skills/harness.md` 의 정본이다. bootstrap 이 ai-ml 종류 선택 시 `§3-B-2` 의 cp 명령으로 사용자 프로젝트의 `.claude/skills/harness.md` (SWE default) 를 이 정본으로 *교체* 한다. 즉 ai-ml 종류 프로젝트에서는 `/harness` 슬래시 호출이 이 워크플로우를 가리킨다 — 별도 `/ml-harness` 슬래시는 존재하지 않는다.

---

## 0. ML 라이프사이클 9 단계 / 3 그룹 / Checkpoint 3 지점

step 설계 전에 이 3 자산을 머리에 두고 phase 를 그룹 단위로 묶는다.

| 그룹 | 단계 | 활동 | 산출물 성격 |
|------|------|------|------------|
| **A** 데이터 분석 | 1. 데이터셋 분석·검증 | 분포·결측·라벨 합의도·PII | `runs/data-sanity-*/report.md` |
| **B** 모델 완성 | 2. 다수 데이터셋·전처리·증강 / 3. 모델(DNN) / 4. loss | model/loss sanity 통과 증빙 | `runs/model-sanity-*/`, `runs/loss-sanity-*/` |
| **C** 실험 | 5. training / 6. 가설→검증→개선 / 7. inference·경량화 / 8. comparison / 9. ablation | `runs/{ts}-{tag}/` 산출물·비교/ablation 표 | `runs/{ts}-{tag}/` (EXPERIMENTS.md 컨벤션) |

**Human-in-the-loop Checkpoint** (사용자 정의 — 3 그룹 사이/안에 사람이 결정해야 하는 자리):

| CP | 위치 | 사용자가 확정 |
|----|------|--------------|
| **CP-1** | A → B 사이 (단계 1 직후) | 데이터셋 그대로 가도 되는지 + 전처리 정책 + class 매핑·좌표계 + **실험 scope/budget** (데이터셋·알고리즘·ablation·시드 → 총 GPU-시간) |
| **CP-2** | B → C 사이 (단계 4 직후) | 모델 아키텍처·파라미터 수·FLOPs·loss·학습 예산 + 첫 phase 의 `hypothesis` |
| **CP-3** | C 중간 (단계 6 의 첫 baseline 직후) | baseline 메트릭 → 다음 분기(inference 경량화 / comparison / ablation) 결정 |

CP 위치의 step 은 완료 후 자동으로 멈춰 사용자 검토를 기다리도록 step 정의에 `checkpoint: "CP-1"` 등으로 표기한다.

---

## 1. 워크플로우

### A. 탐색

다음 docs 를 순서대로 읽는다 (없으면 누락 보고):

- `/CLAUDE.md` (헌법)
- `/docs/PRD.md`, `/docs/ARCHITECTURE.md`, `/docs/ADR.md`
- `/docs/DATA_CARD.md`, `/docs/MODEL_CARD.md`, `/docs/EVAL_PROTOCOL.md`, `/docs/EXPERIMENTS.md`
- `/docs/PRD_VIEW.md` (검수 13 View — 누락 관점 점검)
- `/docs/SCOPE.md` (있으면 — 실험 scope/budget)

읽으며 다음을 머릿속에 정리: **task 종류** (detection/classification/seg/NLP/retrieval/regression 등), **primary metric**, **데이터셋 수**, **알고리즘 후보 수**, **ablation 차원**, **시드 세트**, **per-run 예상 시간**.

### B. 논의

다음 결정이 비어 있으면 사용자에게 한 항목씩 대안 제시 패턴으로 묻는다:

1. **Task 종류와 primary metric** — `eval_spec.task` (detection → mAP@0.5:0.95 / segmentation → mIoU / classification → F1·accuracy / NLP gen → BLEU·ROUGE 등). 라이브러리·임계 해석(↑/↓)도 함께.
2. **실험 scope** — 데이터셋 N·알고리즘 M·ablation 차원 D(values V)·시드 K → 총 학습 횟수 `N × M × Σ(V-1) × K` + per-run 시간 → **총 GPU-시간 예상**. SCOPE.md 가 없으면 한 줄로 만들어 합의.
3. **첫 phase 의 그룹** — A / B / C 중 어디서 시작 (보통 A → B → C).
4. **CP 위치** — phase 마지막 step 이 어느 CP 인지 (또는 CP 없음).
5. **자동화 모드** — `automation: "auto"` (그룹 C 기본) / `"review-each"` (그룹 A·B 기본).

### C. Step 설계

설계 원칙 (일반 harness 의 7 원칙 + ai-ml 보강):

1. **Scope 최소화** — 한 step 은 한 레이어/모듈 또는 *한 실험 셀*.
2. **자기완결성** — 각 step 파일은 독립 세션. "이전 대화에서" 같은 외부 참조 금지.
3. **사전 준비 강제** — 읽을 docs/코드 경로 명시.
4. **시그니처 수준 지시** — 함수/클래스 인터페이스만 제시. 단, **재현성·수치 안정성·평가 정합성** 같은 핵심 규칙은 박는다.
5. **AC 는 실행 가능한 커맨드** — 추상 서술 금지. `python -m src.train ... && jq -e '.f1 >= 0.85' runs/.../eval.json` 같이 *수치 검증* 포함.
6. **주의사항은 구체적으로** — "X 를 하지 마라. 이유: Y" 형식.
7. **네이밍** — kebab-case slug (`data-sanity-cocov2`, `model-fpn-head`, `loss-focal-v2`, `train-baseline-s42`, `bench-int8`, `ablate-aug-strong`).
8. **(추가) 그룹 라벨** — step 의 `kind` 와 phase 의 `group` 을 일관되게.
9. **(추가) 산출물 경로** — `runs/{...}/` 패턴을 step 정의의 `runs_pattern` 으로 명시.
10. **(추가) 메트릭 임계** — 그룹 C step 은 반드시 `success_metric` (jq 표현) 을 명시.

### D. 파일 생성

#### D-1. `phases/index.json` (전체 현황)

SWE `/harness` 와 동일한 스키마 + ai-ml 종류 한정 `experiment_budget` 필드.

```json
{
  "phases": [
    { "dir": "0-data-sanity", "status": "pending", "group": "A" }
  ],
  "experiment_budget": {
    "datasets": 2,
    "algorithms": 3,
    "ablation_dims": [{ "name": "aug", "values": ["none", "weak", "strong"] }],
    "seeds": 3,
    "per_run_hours": 6,
    "total_gpu_hours_estimate": 162
  }
}
```

#### D-2. `phases/{task-name}/index.json` (task 상세)

```json
{
  "project": "<프로젝트명>",
  "phase": "<task-name>",
  "group": "C",
  "automation": "auto",
  "hypothesis": "강한 aug 가 mAP 를 0.02 이상 올린다",
  "baseline_run_id": "20260301-1830-baseline",
  "eval_spec": {
    "task": "detection",
    "metric_primary": "mAP@0.5:0.95",
    "metric_lib": "pycocotools==2.0.7",
    "input_format": { "pred": "xyxy_score_class", "gt": "xyxy_class" },
    "success_threshold": 0.37
  },
  "steps": [
    {
      "step": 0,
      "name": "train-aug-strong-s42",
      "kind": "experiment",
      "timeout_sec": 86400,
      "max_retries": 0,
      "monitors": ["tensorboard:6006", "nvidia-smi:1000"],
      "success_metric": "jq -e '.metric_primary >= 0.37' runs/{run_dir}/eval.json",
      "checkpoint": null,
      "status": "pending"
    }
  ]
}
```

필드 규칙:

- `group`: `"A" | "B" | "C"` — 라이프사이클 그룹.
- `automation`: `"auto"` (그룹 C 권장) / `"review-each"` (그룹 A·B 권장). checkpoint 는 무관하게 항상 멈춤.
- `hypothesis`: 그룹 C phase 는 의무. 한 줄.
- `baseline_run_id`: 비교 대상 run-id (없으면 null).
- `eval_spec`: task 별 메트릭 정의 (없으면 docs/EVAL_PROTOCOL.md 기본).
- `steps[].kind`: `"code" | "data-sanity" | "model-sanity" | "loss-sanity" | "experiment" | "inference-bench" | "quant-bench" | "compare" | "ablate"`.
- `steps[].timeout_sec`: 기본 1800. 학습 step 은 86400 (1 일) 권장.
- `steps[].max_retries`: 기본 3. 학습/실험 step 은 0 권장 (자동 재시도 의미 없음).
- `steps[].monitors`: `["tensorboard:6006", "nvidia-smi:1000"]` 형식. 기본 없음.
- `steps[].success_metric`: jq 표현. 완료 후 자동 평가 (실패 시 status 강제 error).
- `steps[].checkpoint`: `"CP-1" | "CP-2" | "CP-3"` 또는 null. 완료 후 awaiting-review 로 멈춤.

상태 전이 (SWE default 의 4 상태 + ai-ml 신규 3):

| 상태 | 의미 |
|------|------|
| `pending` / `completed` / `error` / `blocked` | 기존 의미 동일 |
| `awaiting-review` | checkpoint step 완료 후 사용자 검토 대기 |
| `approved` | 사용자 검토 통과 → 다음 실행 시 진행 |
| `rejected` | 사용자 검토 거부 → phase 중단 |

`crash_reason`/`crash_evidence`/`recommended_action` 은 step 실패 시 execute.py 가 자동 기록 (스키마 확장 후 활성화). 생성 시 넣지 않는다.

#### D-3. `phases/{task-name}/step{N}.md`

```markdown
# Step {N}: {이름}

## 읽어야 할 파일

- `/docs/ARCHITECTURE.md`
- `/docs/EVAL_PROTOCOL.md`
- `/docs/DATA_CARD.md` (데이터 step) / `/docs/MODEL_CARD.md` (모델·실험 step)
- {이전 step 의 산출물 경로 — runs/.../ 또는 src/...}

이전 step 의 코드/설정/메트릭을 꼼꼼히 읽고 작업하라.

## 작업

{구체적 지시. 시그니처 수준 코드 + 핵심 규칙(재현성·수치 안정성·평가 정합성) 명시.}

## Acceptance Criteria

```bash
{kind 별로 다름 — 아래 §2 참고}
```

## 검증 절차

1. 위 AC 커맨드를 실행.
2. 결과 산출물 (`runs/{...}/{report.md|eval.json|bench.json}`) 확인.
3. `phases/{task}/index.json` 의 해당 step status 갱신 — `completed` + `summary` (실패 시 `error` + `error_message`, 사용자 개입 필요 시 `blocked` + `blocked_reason`).

## 금지사항

- {step 별로 박는 "X 하지 마라. 이유: Y" 형식}
- **가중치 파일(`*.pt`/`*.pth`/`*.ckpt`/`*.safetensors`) 을 git add 하지 마라.** 이유: 레포가 무거워지고 LFS 미설정 시 push 실패.
- **test split 을 학습에 쓰지 마라.** 이유: 누설 시 모든 결과 무효 (EVAL_PROTOCOL.md).
- 기존 테스트를 깨뜨리지 마라.
```

---

## 2. Kind 별 step 패턴 (AC 예시 포함)

각 단계의 AC 는 *수치 검증* 까지 포함해야 한다. "동작한다" 같은 모호한 기준 금지.

### 2-1. `data-sanity` (단계 1 — 그룹 A)

```bash
python -m src.data.sanity --dataset {name} --split-disjoint --pii-scan
# → runs/data-sanity-{ts}/{report.md, stats.json, manifest.json}

# 자동 검증
jq -e '.split_disjoint == true and .pii_violations == 0' runs/data-sanity-{ts}/stats.json
```

검증 항목: train/val/test disjoint, 클래스 분포 표, 결측 비율, **task 별 GT 포맷 검증** (detection 이면 좌표계·class id·box 범위), PII 패턴 매치, 데이터셋 해시 기록.

### 2-2. `model-sanity` (단계 3 — 그룹 B)

```bash
python -m src.models.sanity --model {name} --device cuda
# → runs/model-sanity-{ts}/{report.md, sanity.json}

jq -e '
  .forward_shape_ok == true and
  .all_params_have_grad == true and
  .overfit_one_batch_loss < 0.1 and
  .param_count_m < 100
' runs/model-sanity-{ts}/sanity.json
```

검증 항목: ① forward shape (task 별 — detection `[B,N,6]` / seg `[B,C,H,W]`), ② 파라미터 수, ③ gradient flow (모든 학습 파라미터에 `p.grad` 존재), ④ overfit 1-batch (같은 배치 100 step → loss 임계 이하), ⑤ (선택) FLOPs.

### 2-3. `loss-sanity` (단계 4 — 그룹 B)

```bash
python -m src.loss.sanity --config configs/{loss}.yaml --steps 50
# → runs/loss-sanity-{ts}/{report.md, sanity.json}

jq -e '
  .nan_inf_count == 0 and
  .grad_norm_max < 100 and
  .loss_decreases == true
' runs/loss-sanity-{ts}/sanity.json
```

검증 항목: ① 매 step `assert torch.isfinite(loss)` 통과, ② gradient norm 시계열의 최댓값 임계 이하, ③ 50 step 안에 loss 가 단조 감소 추세, ④ (선택) explosion early-stop 시뮬레이션.

학습 루프에 반드시 박을 것:
- `assert torch.isfinite(loss).all()` — 실패 시 즉시 중단 + `runs/{id}/error.log` 에 배치 인덱스 기록.
- `total_norm = torch.nn.utils.clip_grad_norm_(...)` 의 반환값을 매 step `metrics.csv` 의 `grad_norm` 컬럼으로.

### 2-4. `experiment` (단계 5·6 — 그룹 C)

```bash
python -m src.train --config configs/{cfg}.yaml --seed {S}
# → runs/{YYYYMMDD-HHmm}-{tag}/{config.yaml, git_rev.txt, seed.txt, metrics.csv, checkpoints/, logs/}

python -m src.eval --run-id {YYYYMMDD-HHmm}-{tag} --split test
# → runs/.../eval.json

# 메트릭 임계
jq -e '.metric_primary >= {threshold}' runs/.../eval.json

# baseline 대비 (baseline_run_id 있을 때)
jq -e --argjson base "$(jq .metric_primary runs/{baseline}/eval.json)" \
  '.metric_primary >= $base' runs/.../eval.json
```

산출물 커밋 정책: 코드는 추적, `runs/{id}/{metrics.csv, eval.json, config.yaml, git_rev.txt, seed.txt}` 만 추적, **가중치 미추적**.

권장 step 정의: `timeout_sec: 86400`, `max_retries: 0`, `monitors: ["tensorboard:6006", "nvidia-smi:1000"]`.

### 2-5. `inference-bench` (단계 7 — 그룹 C)

```bash
python -m src.infer.bench --ckpt runs/{id}/checkpoints/best.pt --device cuda \
  --batch-size 1 --warmup 50 --iter 500
# → runs/{id}/bench.json (latency_p50/p95/p99, fps, vram_peak_mb, test_metric)

# 배치 스윕 — batch_size ∈ {1, 4, 16, 64}
python -m src.infer.bench --ckpt ... --batch-sweep "1,4,16,64"

jq -e '.latency_p95_ms < 100 and .vram_peak_mb < 8192' runs/{id}/bench.json
```

### 2-6. `quant-bench` (단계 7 — 그룹 C)

```bash
python -m src.quant.convert --ckpt runs/{id}/checkpoints/best.pt --mode {fp16|int8_ptq|int8_qat|onnx|trt}
# → runs/quant-{id}/{mode}/{ckpt|onnx|engine}

python -m src.infer.bench --ckpt runs/quant-{id}/{mode}/... --device cuda
# → runs/quant-{id}/{mode}/bench.json

python -m src.quant.summary --quant-dir runs/quant-{id}
# → runs/quant-{id}/summary.md (모드 × 지표 표)

# 정확도 drop 2% 이내
jq -e '.metric_drop > -0.02' runs/quant-{id}/{mode}/bench.json
```

### 2-7. `compare` (단계 8 — 그룹 C)

**phase = 한 비교 라운드**. step N 개 = 알고리즘 N 개.

phase 의 `fairness_constraints` 가 동일 데이터셋 split / 시드 세트 / 평가 메트릭 / 하드웨어 / 전처리를 박는다. 모든 step 종료 후 execute.py 가 위반 검사 (스키마 확장 후 활성화).

```bash
# 각 step
python -m src.train --algo {name} --config configs/preprocess.yaml --seed {S}
python -m src.eval --run-id ... --split test

# phase 마지막 step
python -m src.compare.summary --phase compare-{topic}
# → runs/compare-{topic}/{summary_table.md, summary_stats.json}

jq -e '.fairness_violations == 0' runs/compare-{topic}/summary_stats.json
```

### 2-8. `ablate` (단계 9 — 그룹 C)

**phase = 한 ablation 라운드**. step N 개 = M 차원의 조합 N 개.

phase 의 `ablation_dimensions: [{name, values}]` 가 차원 정의. 조합 전략: one-at-a-time (M+1 step) / full factorial / 사용자 지정.

```bash
# 각 step — config 차이만
python -m src.train --config configs/ablate/{config-tag}.yaml --seed {S}
python -m src.eval --run-id ... --split test

# phase 마지막 step
python -m src.ablate.summary --phase ablate-{topic}
# → runs/ablate-{topic}/{summary_table.md, marginal.json}
```

---

## 3. 그룹별 자동화 모드 권장

| 그룹 | `automation` | checkpoint 위치 |
|------|--------------|-----------------|
| **A** 데이터 분석 | `review-each` | 마지막 step 에 **CP-1** |
| **B** 모델 완성 | `review-each` | 마지막 step 에 **CP-2** |
| **C** 실험 | `auto` | 첫 baseline step 에 **CP-3** (이후 step 은 자동) |

`auto` 모드라도 crash 시 사용자 알림 의무 — 죽으면 *왜 죽었는지* 알고 넘어가야 한다.

---

## 4. 실행

```bash
python3 scripts/execute.py {task-name}         # 순차 실행
python3 scripts/execute.py {task-name} --push  # 실행 후 push
```

execute.py 의 ml 종류 자동 처리 (2차-B 까지 활성화):

- `feat-{task-name}` 브랜치 생성/checkout
- **가드레일 주입 — `step.kind` 별 부분집합**:
  - `data-sanity` → `PRD` + `DATA_CARD` 만
  - `model-sanity` → `PRD` + `ARCHITECTURE` + `MODEL_CARD` 만
  - `loss-sanity` → `PRD` + `ARCHITECTURE` 만
  - `inference-bench` / `quant-bench` → `MODEL_CARD` + `EVAL_PROTOCOL` 만
  - `experiment` / `compare` / `ablate` / `code` (또는 누락) → 전체 docs
  - `CLAUDE.md` 는 항상 포함
- 컨텍스트 누적 — 완료된 step 의 summary 를 다음 step 프롬프트에 전달
- step 의 `timeout_sec` (기본 1800) / `max_retries` (기본 3) 자동 적용
- **`runs/` 자동 인덱싱** — step 실행 직전/직후 `runs/` 디렉터리를 비교해 새로 생긴 디렉터리를 step 의 `run_dirs` 에 자동 기록. 각 디렉터리의 `eval.json` 또는 `bench.json` 에서 핵심 메트릭을 `run_metrics` 에 snapshot.
- **`success_metric` 자동 평가** — step 의 `success_metric` jq 표현을 `_index_runs` 의 `run_dirs[0]` 으로 변수 치환 (`{run_dir}` → `runs/<신규 디렉터리>`) 후 평가. 실패 시 status 강제 error, error_message 에 "메트릭 미달" 기록 → 재시도 슬롯 안에서 자가 교정 가능.
- **crash 분류** — step 실패 시 stderr/stdout 을 10 카테고리 패턴 매칭 (OOM / NaN/Inf / DataLoader / GPU/driver / Disk full / SHM / NCCL / Preemption / Hung / Unknown). 분류 결과를 step 의 `crash_reason` / `crash_evidence` (stderr 마지막 20 줄) / `recommended_action` 에 자동 기록.
- **OOM 한정 자동 재시도** — step 정의에 `auto_retry_on_oom: true` 가 있고 첫 crash 가 `OOM` 이면 1 회 추가 시도 (max_retries 와 독립). preamble 에 "batch_size 를 절반으로 줄여 재시도하라" 자동 주입. step 의 `attempted_recovery: {oom_retry: true}` 에 기록.
- **checkpoint step 인터랙티브 prompt** — `step.checkpoint = "CP-1" | "CP-2" | "CP-3"` 가 있으면 completed 직후 stdin 으로 `approve? [y/N]` 묻고 답에 따라 status = `approved` / `rejected` 자동 전이. TTY 가 아니면 (CI 등) `awaiting-review` 로 두고 exit 3 — 사용자가 status 를 직접 수정 후 재실행.
- **monitor 동행 (2차-B)** — `step.monitors: ["tensorboard:6006", "nvidia-smi:1000"]` 가 있으면 step 시작 시 백그라운드 Popen 으로 띄움:
  - `tensorboard:{port}` — `runs/` 전체를 logdir 로, 호스트 `:{port}` 노출. 학습 도중 메트릭 시각화.
  - `nvidia-smi:{interval_ms}` — `runs/gpu.log` 에 GPU util/메모리/온도 시계열 기록.
  - 도구 미설치 환경(non-GPU 등)에선 해당 monitor 만 skip + 경고. step 종료 시 모두 SIGTERM (5초 grace → SIGKILL).
- **heartbeat watchdog (2차-B)** — `step.heartbeat_timeout_sec` 가 있으면 별도 스레드로 `phases/{task}/.heartbeat` mtime 폴링 (5초 간격). 학습 스크립트가 매 N초 touch 안 하면 → 임계 초과 시 subprocess 에 SIGTERM + `crash_reason: "Hung"` 자동 분류. `.heartbeat` 가 한 번도 생성 안 됐으면 학습 초기화 중으로 보고 hang 으로 판단하지 않음.
- **`.claude/settings.json` PreToolUse 패턴 (bootstrap §3-B-3 에서 자동 주입)** — ai-ml 종류 bootstrap 시 SWE default 의 위험 명령 차단(`rm -rf` 등)에 ml 차단 패턴이 OR 로 추가됨: `rm -rf runs/`, `git add *.{pt,pth,ckpt,safetensors}`, `git add -f data/`. 가중치 커밋·실험 결과 통째 삭제·데이터 강제 추가 사고 방지.
- **stdout 실시간 tee** — claude subprocess 의 stdout 을 별도 스레드가 라인 단위로 읽어 `phases/{task}/step{N}.log` 파일에 풀 기록. `step{N}-output.json` 의 `stdout_tail` 에는 마지막 200 줄만. 사용자가 `tail -f phases/{task}/step{N}.log` 로 진행 추적 가능.
- **experiment_budget 사전 검사** — top-level `phases/index.json` 의 `experiment_budget` 가 있으면 run() 진입 시 총 학습 횟수 / GPU-시간 / 디스크 예상 계산. `gpu_hours_threshold` / `disk_gb_threshold` 초과 시 사용자 confirm prompt (TTY 면 [y/N], 아니면 exit 3).
- **공정성 매트릭스 자동 검증** — phase 의 step 중 하나라도 `kind: compare` 또는 `kind: ablate` 면 `_finalize` 에서 모든 `run_dirs` 의 5 차원(dataset_split / git_rev / metric / hardware / seed) 동일 검증. 위반 시 phase status=error + `fairness_violations` 기록.
- **ckpt 기반 재개** — step 정의에 `resumable: true` + `resume_from: "runs/{run_dir}/checkpoints/last.pt"` 가 있으면 재시도 시 preamble 에 "처음부터 다시 학습 말고 ckpt 부터 재개하라" 메시지 + `--resume <ckpt>` 인자 사용 지시 자동 주입.
- **launcher wrap (torchrun)** — step 정의에 `launcher: {kind: "torchrun", nproc_per_node: 4, env: {NCCL_DEBUG: "WARN"}}` 가 있으면 preamble 에 "AC 의 학습 명령을 `torchrun --nproc-per-node=4` 로 wrap 해 실행하라" 지시 자동 주입.
- 2 단계 커밋 (feat / chore)

상태 전이 (확장):

| 상태 | 진입 | 처리 |
|------|------|------|
| `pending` | 초기값 | execute.py 가 다음 실행 대상으로 잡음 |
| `completed` | step 정상 종료 + success_metric 통과 | 다음 step 으로 |
| `error` | crash 발생 또는 메트릭 미달 + 재시도 소진 | execute.py 종료 (exit 1) |
| `blocked` | 사용자 개입 필요 (API 키 등) — Claude 가 직접 기록 | execute.py 종료 (exit 2) |
| `awaiting-review` | checkpoint step 완료 + TTY 미감지 | execute.py 종료 (exit 3) |
| `approved` | checkpoint step 완료 + 사용자 y | 다음 step 진행 |
| `rejected` | checkpoint step 완료 + 사용자 n / 직접 수정 | execute.py 종료 (exit 1) |

에러 복구:

- **error**: `phases/{task}/index.json` 의 step `status` 를 `pending` 으로 + `error_message` 삭제. `crash_reason` 이 `OOM` / `NaN` 이면 `recommended_action` 의 조치 먼저 적용 (batch_size 축소 / lr 낮춤 등).
- **blocked**: `blocked_reason` 해결 후 `status` 를 `pending` 으로 + `blocked_reason` 삭제 후 재실행.
- **awaiting-review**: 검토 자료(`run_dirs` / `run_metrics`) 확인 후 `status` 를 `approved` (진행) 또는 `rejected` (phase 중단) 로 변경 후 재실행.

---

## 5. 학습 스크립트 컨벤션 (execute.py 자동 동작이 의존)

execute.py 의 2차-A 동작은 학습 스크립트가 아래 컨벤션을 지킨다는 전제 위에서 동작한다. 어기면 `success_metric` 검증·`run_metrics` snapshot·OOM 자동 재시도가 정상 작동하지 않는다.

### 5-1. `runs/{id}/` 디렉터리 구조 (필수)

```
runs/{YYYYMMDD-HHmm}-{tag}/
├── config.yaml         # 학습/평가 시점의 하이퍼파라미터 (필수)
├── git_rev.txt         # git rev-parse HEAD (필수)
├── seed.txt            # 시드값 (필수)
├── metrics.csv         # epoch 별 메트릭 시계열 (학습)
├── eval.json           # 평가 결과 — { "metric_primary": <number>, ... } (평가)
├── bench.json          # inference benchmark — { "latency_p95_ms", "vram_peak_mb", "fps", "metric_drop" } (bench)
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── logs/               # tensorboard event / 텍스트 로그
```

`{YYYYMMDD-HHmm}-{tag}` 패턴이 `_snapshot_runs` 의 diff 단위.

### 5-2. `eval.json` / `bench.json` 의 필수 키

execute.py 의 `_extract_run_metric` 가 다음 키만 자동 snapshot 한다. 학습 스크립트는 이 키 이름을 그대로 써야 `run_metrics` 에 잡힌다:

| 파일 | 키 | 의미 |
|------|----|------|
| `eval.json` | `metric_primary` | task 의 1차 메트릭 (mAP, F1, BLEU, mIoU 등 — 숫자 하나) |
| `bench.json` | `latency_p95_ms` | 추론 latency p95 (ms) |
| `bench.json` | `vram_peak_mb` | VRAM peak (MB) |
| `bench.json` | `fps` | throughput (frames/s) |
| `bench.json` | `metric_drop` | quantization 정확도 변화 (음수 = drop) |

추가 키는 자유 — snapshot 안 잡힐 뿐, 파일에는 보존된다.

### 5-3. `success_metric` 의 `{run_dir}` 치환 규칙

step 의 `success_metric` 은 jq 표현이고 `{run_dir}` 변수가 들어가면 `_index_runs` 가 잡은 첫 번째 신규 디렉터리로 치환된다 (`runs/{run_dir}/eval.json` → `runs/20260301-1830-baseline/eval.json`).

```json
{
  "success_metric": "jq -e '.metric_primary >= 0.85' runs/{run_dir}/eval.json"
}
```

jq 가 컨테이너에 설치돼 있어야 한다 (`apt install jq` 또는 dev 이미지 빌드 시 포함). 미설치 시 success_metric 검증은 skip (run_dirs 가 비어 있으면 검증 skip 없이 실패).

### 5-4. `auto_retry_on_oom` 활성화 시 batch_size 인자 컨벤션

execute.py 가 OOM 재시도 시 학습 명령을 직접 수정하지 않는다. preamble 에 "batch_size 를 절반으로 줄여 재시도하라" 메시지를 박고, Claude 가 step.md 의 학습 명령을 다시 실행하면서 batch_size 인자를 조정한다. 이를 위해 학습 명령은 **batch_size 가 한 곳에서 결정**되어야 한다:

| ❌ 피할 패턴 | ✅ 권장 패턴 |
|-------------|-------------|
| `batch_size` 가 `config.yaml` 안에 박혀 외부 인자로 못 바꿈 | `--batch-size 32` (또는 `per_device_train_batch_size`) 같이 CLI 인자로 노출 |
| 여러 곳에 `batch_size` 가 흩어짐 | `train`/`val`/`infer` 모두 같은 인자에서 파생 |

### 5-5. crash 분류가 인식하는 stderr 패턴

execute.py 의 crash_classifier 는 다음 메시지를 stderr 에 보고 분류한다. 학습 스크립트가 같은 패턴을 출력하도록 두면 분류 정확도가 올라간다:

| 카테고리 | 인식 패턴 |
|---------|---------|
| OOM | `CUDA out of memory` / `torch.cuda.OutOfMemoryError` |
| NaN/Inf | `loss is not finite` / `AssertionError.*isfinite` / `NaN.*loss` |
| DataLoader | `FileNotFoundError` / `No such file or directory` / `DataLoader.*Error` |
| GPU/driver | `CUDA error: unspecified launch failure` / `no CUDA-capable device` |
| Disk full | `No space left on device` |
| SHM | `bus error.*DataLoader` (`DataLoader worker ... killed by signal: bus error`) |
| NCCL | `NCCL.*error` / `ncclSystemError` |
| Preemption | exit code 137/143 또는 `SIGTERM/SIGKILL` |
| Hung | execute.py 의 timeout 도달 또는 **heartbeat watchdog 발화** (§5-7 참고) |

학습 루프에 박을 가드 (loss-sanity step 에서 검증):

```python
assert torch.isfinite(loss).all(), "loss is not finite at batch {i}"
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
# total_norm 을 metrics.csv 의 grad_norm 컬럼에 기록
```

### 5-6. checkpoint step 의 인터랙티브 prompt 흐름

```
✓ Step 3: data-sanity-cocov2 [125s]
⏸ [CHECKPOINT CP-1] Step 3 (data-sanity-cocov2) 완료. 검토:
    산출물: 20260301-1820-cocov2
    메트릭: {"20260301-1820-cocov2": {"metric_primary": ...}}
    요약: 분포 / 결측 / 라벨 합의도 / PII 모두 통과
    → 이 step 을 approve 하시겠습니까? [y/N]: y
    ✓ approved (step 3).
```

`y` 면 status=approved + 다음 step 진행, `n` (또는 엔터) 면 status=rejected + phase 중단 (exit 1). TTY 미감지 (CI/cron) 면 awaiting-review 로 두고 exit 3 — 사람이 status 를 수기 수정 후 재실행.

### 5-7. heartbeat watchdog 활용 — `.heartbeat` 파일 컨벤션

학습 step 에 `heartbeat_timeout_sec` 가 정의됐으면 학습 스크립트는 매 N 초 (보통 60초) **`phases/{task}/.heartbeat`** 를 touch 해야 한다. 그러지 않으면 watchdog 가 hang 으로 판단해 SIGTERM 으로 종료시키고 `crash_reason: "Hung"` 을 기록.

학습 루프 권장 패턴:
```python
from pathlib import Path
import time

HEARTBEAT = Path("phases/{task-name}/.heartbeat")  # execute.py 가 phase_dir 기준으로 watch
last_touch = 0.0

for epoch in range(epochs):
    for batch in dataloader:
        ...  # forward / backward / step
        now = time.time()
        if now - last_touch >= 60:
            HEARTBEAT.touch()
            last_touch = now
```

watchdog 의 의미:
- **`.heartbeat` 파일이 한 번도 생성 안 됐으면** — 학습 초기화 (데이터 로드, 모델 초기화) 중으로 보고 hang 으로 판단 안 함.
- **한 번이라도 touch 된 뒤 timeout_sec 동안 갱신 없으면** — deadlock / GPU hang / 외부 응답 대기 의심 → SIGTERM 으로 종료, crash_reason=Hung.
- `heartbeat_timeout_sec` 미설정 step 은 watchdog 안 띄움 (기존 timeout 만).

권장값: 학습 1 epoch 의 평균 시간 × 2 (예: epoch 5 분이면 600 초). 너무 짧으면 false positive (정상 학습인데 hang 으로 잡힘).

### 5-8. ckpt 재개 — `--resume` 인자 컨벤션

step 정의에 `resumable: true` 면 학습 스크립트는 다음 인자를 받아야 한다:

```bash
python -m src.train --config configs/{cfg}.yaml --seed 42 --resume runs/{id}/checkpoints/last.pt
```

execute.py 가 재시도 시 preamble 에 ckpt 경로를 박아 Claude 에게 위 형태로 실행 지시. 학습 스크립트는 `--resume` 인자가 주어지면 ckpt 의 epoch/optimizer state/scheduler state 까지 복원해야 함. 단순히 weight 만 로드하면 학습 곡선이 깨짐.

### 5-9. launcher (torchrun) 컨벤션

step 정의에 `launcher: {kind: "torchrun", nproc_per_node: N}` 가 있으면 execute.py 가 preamble 에 wrap 지시 박음. step.md 의 AC 안 학습 명령은 *launcher 없이* 작성하고, execute.py 의 자동 주입에 맡긴다:

```bash
# step.md 의 AC — launcher 없는 raw 명령
python -m src.train --config configs/{cfg}.yaml --seed 42

# execute.py 가 Claude 에게 박는 wrap 지시
torchrun --nproc-per-node=4 python -m src.train --config configs/{cfg}.yaml --seed 42
```

이렇게 두면 같은 step 을 single-GPU 환경에서 돌릴 때 `launcher` 필드만 빼면 됨.

### 5-10. experiment_budget — top-level phases/index.json 컨벤션

`phases/index.json` 의 task 레벨에 `experiment_budget` 필드 추가:

```json
{
  "phases": [...],
  "experiment_budget": {
    "datasets": 2, "algorithms": 3,
    "ablation_dims": [{"name": "aug", "values": ["none", "weak", "strong"]}],
    "seeds": 3,
    "per_run_hours": 6, "per_run_disk_gb": 12,
    "gpu_hours_threshold": 200,
    "disk_gb_threshold": 500
  }
}
```

execute.py 가 run() 진입 시 `total_runs = datasets × algorithms × ∏|values| × seeds`, `total_gpu_hours = total_runs × per_run_hours`, `total_disk_gb = total_runs × per_run_disk_gb` 계산. 임계 초과 시 사용자 confirm 또는 exit 3.

### 5-11. fairness_constraints — compare/ablate phase 의 step 산출물 의무

`kind: compare` 또는 `kind: ablate` step 의 run_dir 에는 다음 5 자산이 모두 있어야 한다:

| 파일 | 내용 | 비교 차원 |
|------|------|---------|
| `config.yaml` 의 `data:` 섹션 (또는 `data_manifest.json`) | 데이터셋 split + 비율 + hash | dataset_split_hash |
| `seed.txt` | 시드값 | seed |
| `git_rev.txt` | git rev-parse HEAD | git_rev |
| `eval.json` 의 `metric_primary` (또는 `metric_primary_name`) | 평가 메트릭 이름 | metric_primary_key |
| `config.yaml` 의 `hardware:` 또는 `bench.json` 의 `hardware` | 하드웨어 태그 | hardware_tag |

`_finalize` 의 fairness verify 가 위 차원이 모든 step run_dir 에 걸쳐 동일한지 확인. 다르면 phase=error.

### 5-12. monitor 동행 활용

step 정의의 `monitors: ["tensorboard:6006", "nvidia-smi:1000"]` 가 있으면 execute.py 가 학습 시작과 함께 백그라운드로 띄움.

- **tensorboard** — `runs/` 전체를 logdir 로 본다. 학습 스크립트는 `runs/{id}/logs/` 에 event 파일을 쓰면 자동으로 dashboard 에 노출됨. 호스트 `http://localhost:6006` 에서 접속 (docker-compose 의 dev 서비스에 6006 포트 노출 필요).
- **nvidia-smi** — `runs/gpu.log` 에 timestamp, GPU util, memory, temp 시계열 기록. 학습 후 분석 자료.
- 도구 없는 환경 (tensorboard 미설치 / GPU 없음) 에선 해당 monitor 만 skip + 경고. 학습 자체는 그대로 진행.
- step 종료 시 모두 SIGTERM (5초 grace → SIGKILL).

---

## 6. 금지사항

- **본 정본을 SWE 종류 프로젝트에 수동 복사하지 마라.** bootstrap §3-B 가 ai-ml 종류일 때만 자동 배치한다. SWE 종류의 `/harness` 는 별도 SWE default 정본 (루트 `.claude/skills/harness.md`) 으로 동작.
- **메트릭 임계 없이 그룹 C step 을 만들지 마라.** `success_metric` 누락 시 baseline 비교가 후행 사람 판단에 의존하게 된다.
- **시드를 step 안에 하드코딩하지 마라.** 모두 `--seed` 인자로. (EXPERIMENTS.md 의 시드 정책)
- **가중치/runs 산출물 통째를 git add 하지 마라.** `metrics.csv / eval.json / config.yaml / git_rev.txt / seed.txt` 만.
- **CP step 의 `automation` 을 무력화하지 마라.** 그룹 C 의 `auto` 모드여도 CP-3 는 사람이 본다.
- **fairness_constraints 우회를 위해 compare/ablate step 의 시드/데이터를 다르게 두지 마라.** 결과 무효.
- **fairness 검증을 우회하지 마라.** compare/ablate phase 의 step 마다 시드/데이터/하드웨어를 다르게 설정하면 `_finalize` 의 검증에서 잡혀 phase=error 가 된다. 검증을 끄는 옵션은 의도적으로 제공하지 않는다.
- **experiment_budget 의 임계를 임의로 늘리지 마라.** 임계 초과 시 confirm prompt 가 뜨는 것은 사용자 결정을 강제하는 안전판이다. 임계를 무조건 큰 값으로 박으면 budget 자체가 무의미.
- **launcher 가 있는 step 의 AC 안에 직접 `torchrun` 을 박지 마라.** execute.py 의 자동 wrap 과 충돌해 `torchrun torchrun ...` 가 된다. AC 는 raw `python -m ...` 만, launcher 는 step 정의의 `launcher` 필드로.
- **resumable step 의 학습 스크립트가 `--resume` 무시하고 처음부터 학습하면 안 된다.** epoch/optimizer state/scheduler state 까지 복원 필수. 단순 weight load 만 하면 학습 곡선이 깨져 ckpt 재개의 의미가 사라진다.
- **본 정본도 SWE default 와 같이 LLM_GUIDE 4 원칙 (Think Before Coding / Simplicity First / Surgical Changes / Goal-Driven Execution) 을 따른다.**
