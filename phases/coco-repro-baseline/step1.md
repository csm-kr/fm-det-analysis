# Step 1: compare-to-baseline

## 읽어야 할 파일

- `/docs/PRD.md` — 성공 기준 (AP 46.2 ± 0.5)
- `/docs/EXPERIMENTS.md` — 단계적 FM 전환 로드맵
- `/runs/{ts}-coco-repro-baseline/eval-*/eval.json` — step 0 산출

## 작업

step 0 의 eval 결과를 본 repo (DiffusionDet AP 46.2) 와 비교해 `runs/coco-repro-baseline-compare/summary.md` 작성.

내용:
- 본 repo 의 baseline 한 줄 (AP 46.2, AP50 65.9, AP75 50.1 등)
- 본 학습의 eval.json 표 (mAP, AP50/75, APs/m/l)
- 차이 ΔAP (signed) 와 ±0.5 임계 통과 여부
- 차이가 의미 있다면 가능한 원인 (epoch / LR / aug 등) 3 줄

## Acceptance Criteria

```bash
test -f runs/coco-repro-baseline-compare/summary.md
```

## 검증 절차

1. summary.md 작성.
2. `phases/coco-repro-baseline/index.json` step 1 status → completed.
3. PENSIEVE 의 "최근 ablation / 진단 결과" 표에 한 줄 mirror.

## 금지사항

- summary.md 안에 가중치 / runs/ 내부 파일 경로 절대 경로로 박지 마라. 상대 경로 권장.
- ΔAP 가 +0.5 이내라고 P0a 진단을 skip 하지 마라 (PRD 성공 기준의 두 번째 부분: 진단 5행).
