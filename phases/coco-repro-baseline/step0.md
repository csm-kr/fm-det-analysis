# Step 0: train-coco-baseline-s42 (CP-3 후행)

## 읽어야 할 파일

- `/CLAUDE.md` — 시드 강제 / detectron2 금지 / 평가 프로토콜 변경 금지 (CRITICAL)
- `/docs/PRD.md` — 성공 기준 (P0 AP 46.2 ± 0.5)
- `/docs/EVAL_PROTOCOL.md` — 800~1333 / iter step 4 / num_eval_boxes 500
- `/configs/experiment/coco-repro-baseline.yaml`
- `/configs/train/baseline.yaml` — AdamW(2.5e-5) + MultiStepLR(47,57) + AMP + grad_clip 1.0
- `/train.py` / `/eval.py`

## 작업

DiffusionDet 본 repo (COCO val AP 46.2) 재현 학습. 1 GPU (RTX PRO 6000 Blackwell 48GB) 단일.

**예상 시간**: 50-125 hour (2-5 day) — iter 0.4-1.0s 가정, 61 epoch × 7,392 iter/epoch ≈ 450k iter.

**명령**:

```bash
# 학습 (백그라운드, nohup)
nohup env TORCH_HOME=/workspace/fm-det/.cache/torch \
  python train.py +experiment=coco-repro-baseline seed=42 \
  > phases/coco-repro-baseline/train.log 2>&1 &
echo $! > phases/coco-repro-baseline/train.pid

# 모니터링
tail -f phases/coco-repro-baseline/train.log
nvidia-smi
tail -f runs/*-coco-repro-baseline/metrics.csv

# 중단 후 재개 (학습 끊겨서 last.pt 로 부터)
python train.py +experiment=coco-repro-baseline seed=42 \
  +train.resume=runs/{ts}-coco-repro-baseline/checkpoints/last.pt
# (NOTE: train.py 가 --resume 인자 미구현 시 추가 필요)

# 학습 끝나면 평가
python eval.py +experiment=coco-repro-baseline seed=42 \
  run_dir=runs/{ts}-coco-repro-baseline
# → runs/{ts}-coco-repro-baseline/eval-{HHmm}/eval.json
```

산출:
- `runs/{ts}-coco-repro-baseline/{config.yaml, git_rev.txt, seed.txt}`
- `runs/{ts}-coco-repro-baseline/metrics.csv` (epoch 별 loss/grad_norm/lr)
- `runs/{ts}-coco-repro-baseline/checkpoints/{last.pt}` (매 epoch)
- 학습 완료 후 `eval.py` 호출 → `runs/{ts}-coco-repro-baseline/eval-{HHmm}/eval.json` (metric_primary=AP)

## Acceptance Criteria

```bash
# 학습 완료 + 평가 산출 후
jq -e '.metric_primary >= 0.457 and .metric_primary <= 0.467' \
  runs/{ts}-coco-repro-baseline/eval-*/eval.json
```

success_metric 미달 (AP < 45.7) 이면 I-04 의 "재현 미달" — 3-seed × 향상 요소 ablation 으로 보강.
AP > 46.7 도 비정상 (baseline 초과는 평가 프로토콜 의심).

## 검증 절차

1. 학습 명령 백그라운드 시작 → 1-2분 내 metrics.csv 에 첫 iter 기록 확인.
2. 매 epoch 끝 (약 1-2h) ckpt 저장 + 로그 한 줄 확인.
3. 61 epoch 후 학습 자연 종료.
4. eval.py 호출 → eval.json 산출.
5. `phases/coco-repro-baseline/index.json` step 0 status: pending → completed + summary.

## 금지사항

- **--seed 인자 생략 금지** (CLAUDE.md CRITICAL). 학습 시작 차단.
- **평가 프로토콜 변경 금지** — short_sides=[800], max_size=1333, num_proposals=500, inference_steps=4 모두 본 repo 동치.
- **`runs/{id}/checkpoints/last.pt` 를 git add 하지 마라** (가중치 차단 hook).
- **OOM 시 batch_size 를 늘리지 마라** (재현 무효). 작게 줄여 재개.
- **AMP 첫 step 의 grad_norm=NaN 은 정상** — scaler 가 inf grad 감지해 step skip.
- 학습 중 train.py 코드 변경 금지 — `git_rev.txt` 와 실제 학습 코드가 어긋남.
