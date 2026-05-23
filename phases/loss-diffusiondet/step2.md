# Step 2: loss-sanity-50step

## 읽어야 할 파일

- `/docs/ARCHITECTURE.md` — DiffusionDet loss 트리
- `/losses/criterion.py` — SetCriterion (focal × class_weight, L1 × l1_weight, GIoU × giou_weight, deep supervision K=6)
- `/losses/matcher.py` — SimOTA dynamic-k
- `/losses/sanity.py` — sanity 스크립트 (이미 작성됨)
- `/configs/loss/diffusion.yaml`

## 작업

DiffusionDet 의 set loss (focal + L1 + GIoU, 6 layer deep supervision) 가 50 step 학습 안에서:
- NaN / Inf 가 발생하지 않고
- grad norm 이 explosion 천장 안 (clip 10.0 적용 전 raw norm 도 100k 미만)
- loss 가 *상대 감소* (last < first × 0.8)

세 신호를 모두 만족하는지 검증한다.

`losses/sanity.py` 는 다음을 측정:
- nan_inf_count — 매 step `torch.isfinite(loss).all()` 실패 횟수
- grad_norm_max — `clip_grad_norm_` 의 반환값 (clip *전* total_norm) 의 50-step 최댓값
- loss_decreases — `losses[-1] < losses[0] * 0.8`

**산출**: `runs/loss-sanity-{YYYYMMDD-HHMM}/{sanity.json, report.md}`

**실행 절차**:

```bash
TORCH_HOME=/workspace/fm-det/.cache/torch python -m losses.sanity \
  --steps 50 --batch-size 2 --seed 42 --lr 1e-3 --grad-clip 10.0 --device cuda
```

- TORCH_HOME = workspace 안 캐시 (I-07 workaround).
- 약 30-60s 소요.

## Acceptance Criteria

```bash
ls runs/loss-sanity-*/sanity.json
jq -e '
  .nan_inf_count == 0
  and .grad_norm_max < 100000
  and .loss_decreases == true
' runs/loss-sanity-*/sanity.json
```

## 검증 절차

1. `python -m losses.sanity --steps 50 ...` 실행.
2. `runs/loss-sanity-{ts}/sanity.json` 의 값 확인 — `nan_inf_count=0`, `grad_norm_max < 100000`, `loss_decreases=true`.
3. `report.md` 의 loss / grad_norm trace 가 흐름상 reasonable 한지.
4. `phases/loss-diffusiondet/index.json` 의 step 2 status → `completed` + `summary`.

## 금지사항

- **`grad_norm_max < 100` 같은 절대 임계를 다시 쓰지 마라.** 이유: random init 의 grad norm 은 수백~수만 범위가 정상 — `< 100` 은 학습 시작도 막는 임계. DiffusionDet 본 repo 도 clip 10.0 사용. 본질은 NaN/Inf 부재 + 단조 감소.
- **`torch.isfinite(loss)` assertion 을 sanity 안에서 빼지 마라.** 이유: NaN loss 가 발생하면 즉시 중단해야 무한 backward 가 메모리 폭발하지 않음 (harness §5-5 crash 패턴 인식).
- 가중치 / runs/ 산출물 통째 git add 금지.
