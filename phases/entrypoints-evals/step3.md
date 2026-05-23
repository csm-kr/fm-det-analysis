# Step 3: dry-run-1iter

## 읽어야 할 파일

- `/train.py` — Hydra @main + max_iters 인자
- `/configs/experiment/coco-repro-baseline.yaml`
- `/configs/data/coco.yaml` — `data/coco/{annotations,train2017}` 경로

## 작업

train.py 가 end-to-end 동작 (dataset 로드 → forward → loss → backward → grad clip → optimizer step → metrics.csv 한 줄) 하는지 1-iter 학습으로 검증.

```bash
TORCH_HOME=/workspace/fm-det/.cache/torch python train.py \
  +experiment=coco-repro-baseline seed=42 +train.max_iters=1 tag=dry-run
```

산출: `runs/{YYYYMMDD-HHmm}-dry-run/` — 안에 `config.yaml`, `git_rev.txt`, `seed.txt`, `metrics.csv` (1 row + header), `checkpoints/last.pt`.

AMP 첫 step 의 `grad_norm=NaN` 은 정상 — `GradScaler` 가 inf grad 를 감지해 step 을 skip 함. 다음 iter 부터 정상화.

## Acceptance Criteria

```bash
ls runs/*-dry-run/metrics.csv | head -1   # 최소 1 개 산출
# (success_metric: test -f runs/*-dry-run/metrics.csv 의 glob 변형)
```

## 검증 절차

1. 명령 실행, `runs/{ts}-dry-run/` 생성 확인.
2. `cat runs/{ts}-dry-run/metrics.csv` — header + 1 row + loss_total 가 유한.
3. `runs/{ts}-dry-run/checkpoints/last.pt` 존재.
4. `phases/entrypoints-evals/index.json` step 3 status → `completed`.

## 금지사항

- **dry-run 산출(`runs/*-dry-run/last.pt`) 을 git add 하지 마라.** 이유: 가중치 차단 + .gitignore `runs/`.
- **max_iters 가 없을 때 정식 학습이 1 iter 만 돌도록 default 를 바꾸지 마라.** 이유: P0 학습이 1 iter 만 돌면 학습 자체가 안 됨.
- AMP scaler 의 grad_norm=NaN 첫 step 을 error 로 처리하지 마라 (정상 동작).
