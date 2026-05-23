# Step 2: entrypoints-train-eval-infer

## 읽어야 할 파일

- `/CLAUDE.md` — CRITICAL (특히 시드 강제, detectron2 금지, 평가 프로토콜)
- `/docs/ARCHITECTURE.md` — 진입점 트리
- `/configs/train.yaml` / `/configs/eval.yaml` / `/configs/train/baseline.yaml`
- `/configs/experiment/coco-repro-baseline.yaml`
- `/models/diffusiondet.py` / `/losses/criterion.py`
- `/evals/coco.py` / `/evals/voc.py`

## 작업

루트에 3 진입점을 작성한다. 모두 **Hydra `@main(version_base=None, config_path="configs")`** + `seed` 강제 (없으면 에러).

### `/train.py` — `config_name="train"`
- 시드 고정: random / numpy / torch / cuda + `cudnn.deterministic=True`, `benchmark=False`.
- `runs/{output_dir}/` 산출 — `config.yaml` / `git_rev.txt` / `seed.txt` / `metrics.csv` / `checkpoints/last.pt`.
- AdamW (`cfg.train.optimizer.lr=2.5e-5`, `weight_decay=1e-4`) + MultiStepLR(milestones=[47,57], gamma=0.1).
- AMP (`torch.amp.GradScaler("cuda")`) + grad_clip 1.0.
- 학습 루프: `assert torch.isfinite(loss).all()` + grad_norm 측정 후 metrics.csv 한 줄.
- **dry-run 인자**: `train.max_iters > 0` 이면 그 횟수만 돌고 종료 (step 3 에서 사용).

### `/eval.py` — `config_name="eval"`
- `run_dir` / `ckpt` 인자 필수 (Hydra `run_dir=...` override).
- ckpt 로드 (없으면 random init 경고) → `evals.coco_eval` 또는 `evals.voc_eval` 호출 → `runs/{run_dir}/eval-{HHmm}/eval.json` 산출 (key `metric_primary`).

### `/infer.py` — `config_name="eval"` (eval config 공유)
- `image=path` 인자 필수.
- transform 적용 → model.eval() → top-K (score ≥ `score_thresh`) detections → predictions.json + drawn.jpg.

## Acceptance Criteria

```bash
test -f train.py && test -f eval.py && test -f infer.py
python3 -c 'import train, eval, infer'  # 또는 단순히 importable 한지
```

## 검증 절차

1. 세 파일 존재 + import 성공.
2. `python train.py --help` 가 Hydra 도움말을 보여줌.
3. step 3 (dry-run-1iter) 의 1-iter 학습이 동작.

## 금지사항

- **`import detectron2` 금지.** 이유: CLAUDE.md CRITICAL.
- **seed 가 `???` 또는 None 일 때 그냥 진행하지 마라.** 이유: ablation 결과 재현 불가. 학습 시작 차단.
- **AMP 활성 시 `scaler.unscale_(optim)` 누락하지 마라.** 이유: grad_clip 이 unscaled grad 에 적용돼야 의미. scale 된 grad 에 clip 하면 effective clip 이 잘못됨.
- **가중치(`*.pt`) 를 git add 하지 마라.**
