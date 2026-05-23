# Step 3: model-sanity-overfit (CP-2)

## 읽어야 할 파일

- `/docs/ARCHITECTURE.md` — DiffusionDet 모듈 트리
- `/docs/MODEL_CARD.md` — 변종 카탈로그 + 메커니즘 진단 슬롯
- `/models/diffusiondet.py` — train mode 출력 `[B, K=6, N, C]` / `[B, K=6, N, 4]`
- `/models/sanity.py` — sanity 스크립트 (이미 작성됨)
- `/configs/model/diffusiondet.yaml` / `/configs/loss/diffusion.yaml`

## 작업

DiffusionDet 모델이 정상으로 train forward + backward + parameter update 를 수행하는지 검증한다. harness §2-2 의 model-sanity 4 항목 검증 + DiffusionDet 의 set loss 특성 (absolute 가 커서 *상대 감소* 로 평가) 반영.

**1 batch overfit 패턴**:
- batch_size=2, image 800×800, 무작위 GT 2 개/이미지
- 같은 batch 를 200 step AdamW(lr=1e-3) 로 학습
- 200 step 후 loss 가 `first × 0.8` 미만이면 backward 가 동작한다는 신호

**산출**: `runs/model-sanity-{YYYYMMDD-HHMM}/{sanity.json, report.md}`

sanity.json 의 필수 키:
- `forward_shape_ok` (bool) — pred_logits 가 `[B, 6, 500, 80]`, pred_boxes 가 `[B, 6, 500, 4]`
- `all_params_have_grad` (bool) — trainable param 모두에 `p.grad is not None`
- `param_count_m` (float) — 전체 파라미터 (M 단위)
- `loss_decreased` (bool) — `last_loss < first_loss × 0.8`
- `loss_drop_ratio`, `first_loss`, `last_loss`, `losses_first10`, `losses_last10`, `pred_*_shape`, `device`, `seed`, `steps`, `batch_size`, `image_hw`

**실행 절차**:

```bash
TORCH_HOME=/workspace/fm-det/.cache/torch python -m models.sanity \
  --steps 200 --batch-size 2 --seed 42 --lr 1e-3 --device cuda
```

- TORCH_HOME = workspace 안 캐시 (I-07 workaround — torch-cache named volume 영구화 깨짐).
- 약 60-90s 소요 (GPU forward 200 회).

## Acceptance Criteria

```bash
# 산출 디렉터리
ls runs/model-sanity-*/sanity.json

# success_metric (jq 표현)
jq -e '
  .forward_shape_ok == true
  and .all_params_have_grad == true
  and .param_count_m < 150
  and .loss_decreased == true
' runs/model-sanity-*/sanity.json
```

## 검증 절차

1. `python -m models.sanity --steps 200 --batch-size 2 --seed 42 --lr 1e-3 --device cuda` 실행.
2. `runs/model-sanity-{ts}/sanity.json` 의 값 확인 — `forward_shape_ok=true`, `all_params_have_grad=true`, `param_count_m ≈ 110.7`, `loss_drop_ratio ≥ 0.2`.
3. `report.md` 한 번 훑기 — loss trace 가 단조 감소 추세인지.
4. `phases/model-diffusiondet/index.json` 의 step 3 status → `completed` + `summary` 한 줄 (산출 디렉터리 / 핵심 값). checkpoint=CP-2 라 execute.py 가 사용자 검토 prompt 띄움.

## 금지사항

- **GT 좌표를 image 범위 밖으로 두지 마라.** 이유: `box_xyxy_normalize` 에서 음수 / NaN 유발. sanity 의 무작위 GT 도 `clamp(0,1)` 적용 후 image 좌표로 denormalize.
- **`overfit_one_batch_loss < 1.0` 같은 절대 임계를 다시 쓰지 마라.** 이유: DiffusionDet set loss 의 absolute 는 학습 끝에도 5-15 — `< 1.0` 은 비현실. 학습 동작 신호는 *상대 감소* (`loss_decreased`).
- **GPU 미존재 시 CPU 폴백을 자동 적용하지 마라.** 이유: 본 step 의 의의는 GPU forward + backward 검증 (Blackwell sm_120 후속 신뢰성). CPU 가 필요하면 `--device cpu` 인자 명시.
- 가중치 파일 (`*.pt` 등) 을 git add 하지 마라.
