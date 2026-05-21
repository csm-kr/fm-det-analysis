# Evaluation Protocol

> **CRITICAL — CLAUDE.md**: 본 프로토콜의 모든 항목(데이터 분할, 메트릭 정의, 평가 조건, baseline)은 **변경 금지**. 변경 시 모든 이전 ablation 결과가 무효가 된다. 변경이 정말 필요한 경우 ADR 신설 + 모든 결과 재실행이 의무.

## 평가 데이터
fm-det 의 모든 변종은 **학습한 데이터셋에 대응하는 평가 분할** 에서만 측정한다. 데이터셋 cross-evaluation 없음.

- **COCO 학습 변종**: `val2017` (5,000 장) 에서 평가.
- **VOC 학습 변종**: `VOC2007 test` (4,952 장) 에서 평가.
- 학습/검증과 절대 분리. test 누설 시 모든 결과 무효 (DATA_CARD.md 원칙 인용).
- 외부 비교용 공개 벤치마크:
  - **COCO**: DiffusionDet 본 repo (`DiffusionDet/README.md` 의 표 4행).
  - **VOC**: DiffusionDet 본 repo 가 VOC 결과 미공개 → **본 프로젝트 내 자체 baseline (P0-voc 재구현치) 만 비교 기준**.

## 메트릭

### COCO (주 지표)
| 이름 | 정의 | 기준값 (baseline 매칭) |
|------|------|---------------------|
| AP (mAP@[0.5:0.95]) | COCO 공식 — 10 IoU 임계 평균 (`pycocotools.cocoeval`) | ≥ 45.7 (본 repo "wo box renewal") <br>≥ 46.2 (본 repo full, **재현 목표**) |
| AP50 | IoU=0.5 에서의 AP | 보조 — baseline 매칭 후 변동만 기록 |
| AP75 | IoU=0.75 에서의 AP | 보조 |
| APs / APm / APl | 작은/중간/큰 객체별 AP | 슬라이스 분석 |
| FPS | 1 GPU, batch=1, 800x1333 입력 1장 처리 시간의 역수 | 보조 (DiffusionDet 본 repo 미공개 — 본 프로젝트 측정치만 비교) |

### VOC (동등 위상, 별도 baseline)
| 이름 | 정의 | 기준값 |
|------|------|--------|
| mAP@0.5 | VOC 공식 — IoU=0.5, 11-point AP | 본 프로젝트 P0-voc 재구현치 ± 0.5 (자체 baseline, 외부 SOTA 비교 없음) |

### ablation 결론
- 임계: **ΔAP ≥ 0.5 (COCO)** 이고 **3-시드 평균 std 의 2배 이상** 이어야 "의미 있는 차이" 로 본다. 이하는 노이즈 처리.

## 평가 조건 (변경 금지)
| 항목 | 값 | 근거 |
|------|----|------|
| 입력 해상도 | 짧은 변 800 / 긴 변 최대 1333 (DiffusionDet 동일) | DiffusionDet 본 repo |
| iter step (denoising / FM ODE step) | 4 | DiffusionDet 본 repo 의 main 행 |
| num eval boxes | 500 | DiffusionDet 본 repo 의 main 행 |
| NMS | DiffusionDet 본 repo 와 동일 IoU 임계 (0.5) | 본 repo |
| 전처리 | mean/std 정규화 = DiffusionDet 본 repo 동일 | 본 repo |

위 조건은 baseline (DiffusionDet 본 repo) 와 동일 — fm-det 의 모든 변종이 동일하게 적용. ablation 에서 이 조건 자체는 변경하지 않는다.

## Baseline (변경 금지)
- **D-paper**: DiffusionDet 원 논문 (iter 4 / boxes 500 / AP 46.8) — 최상위 참조.
- **D-repo**: 본 프로젝트의 레퍼런스 `DiffusionDet/` 재현치 (iter 4 / boxes 500 / **AP 46.2**) — fm-det 의 1차 재현 목표.
- **D-repo-noBR**: 본 repo "wo box renewal" 변종 (AP 45.7) — 보조 참조.

동일 평가 분할(`val2017`)에서 측정. baseline AP 는 시점에 따라 변경되지 않으므로 본 문서에 박제.

## 실행 방법
```bash
# Hydra 진입점 — +experiment 로 학습 시점의 config 그대로 로딩, run_dir 로 가중치 위치 지정
python eval.py +experiment=coco-repro-baseline run_dir=runs/{YYYYMMDD-HHmm-tag} split=coco_val
python eval.py +experiment=voc-repro-baseline  run_dir=runs/{YYYYMMDD-HHmm-tag} split=voc_test
# 결과는 runs/{id}/eval.json 에 저장 — MODEL_CARD.md / EXPERIMENTS.md 가 인용
```

`eval.json` 형식:
```json
{
  "split": "coco_val",
  "n_images": 5000,
  "AP": 46.21, "AP50": 65.8, "AP75": 50.3,
  "APs": 28.4, "APm": 49.2, "APl": 63.1,
  "FPS": 11.3,
  "iter_step": 4, "num_eval_boxes": 500,
  "seed": 42, "git_rev": "abc1234"
}
```

## 통계적 유의성
- **단일 시드 (default)**: 모든 ablation 행은 우선 시드 42 단일 실행으로 측정.
- **3-시드 재평가**: ΔAP ≥ 0.5 (또는 결론 후보) 인 변종은 시드 42, 17, 2024 로 재실행 → 평균 ± 표준편차 보고.
- baseline 과의 차이는 **3-시드 평균의 차이가 표준편차의 2배 이상** 일 때만 "의미 있는 개선" 으로 보고. (간이 z-test, 시드 수 적어 정식 t-test 대신.)
- 오차 범위 안의 차이는 EXPERIMENTS 표의 "결론 한 줄" 컬럼에 **"노이즈 — 차이 없음"** 명시.

## 슬라이스 분석 (Sub-group)
주 메트릭(AP) 외에 다음 슬라이스로 편향·실패 모드를 추적:
- **객체 크기**: COCO 의 APs (≤32²) / APm (32²~96²) / APl (≥96²) — 표준 COCO 산출.
- **클래스 long-tail**: AP per category (상위 20 / 하위 20 클래스 별도 표) — `pycocotools` 의 `params.catIds` 분리.
- **iter step 민감도**: iter step ∈ {1, 2, 4, 8} 에서 AP 변화 — sampler 변경 변종(M1, M2) 에 한해 측정.

## 리포트
- `runs/{id}/eval.json` 생성 의무.
- 본 표의 "평가 결과 요약" 은 [MODEL_CARD.md](./MODEL_CARD.md) 에 반영.
- ablation 한 행은 [EXPERIMENTS.md](./EXPERIMENTS.md) 의 표에 자동 append (구현은 P3 phase).

## 관련 문서
- 결정 근거: [ADR.md](./ADR.md)
- 실험 운영: [EXPERIMENTS.md](./EXPERIMENTS.md)
- 모델 카드: [MODEL_CARD.md](./MODEL_CARD.md)
- 데이터셋: [DATA_CARD.md](./DATA_CARD.md)
