# Data Card

## 데이터셋 개요
fm-det 은 공개 학술 detection 벤치마크 **COCO 2017** 과 **PASCAL VOC 07+12** 를 사용한다. 두 데이터셋은 **각각 독립적으로 학습·평가**한다 — 합본 학습하지 않으며, 한 데이터셋에서 학습한 모델을 다른 데이터셋에서 평가하지도 않는다 (cross-evaluation 범위 외). baseline 비교 신뢰도(CRITICAL — 평가 프로토콜 변경 금지)를 위해 두 데이터셋의 **원본 분할을 변형하지 않는다**.

| 데이터셋 | 출처 | 라이선스 | 취득 방법 | 규모 |
|---------|------|---------|----------|------|
| COCO 2017 | https://cocodataset.org/ | CC-BY 4.0 (이미지) / 자체 (annotation) | 공식 `wget` 스크립트 | ≈ 118K train + 5K val, 19GB |
| PASCAL VOC 07 | http://host.robots.ox.ac.uk/pascal/VOC/voc2007/ | "flickr terms" | 공식 tar | ≈ 5K trainval + 5K test, 0.4GB |
| PASCAL VOC 12 | http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ | "flickr terms" | 공식 tar | ≈ 11K trainval, 2GB |

다운로드 진입점은 `datasets/coco/download.py` (ARCHITECTURE.md 의 datasets/ 모듈 정책). 호출: `python -m datasets.coco.download --target coco --split val --seed 42`. SHA256 체크섬 + ZIP CRC + 파일 개수로 무결성 검증 — `runs/data-sanity-download-{ts}/manifest.json` 자동 기록.

## 디렉터리 레이아웃
```
data/                                  # .gitignore
├── coco/
│   ├── annotations/{instances_train2017,instances_val2017}.json
│   ├── train2017/{*.jpg}
│   └── val2017/{*.jpg}
└── voc/
    ├── VOC2007/{Annotations,ImageSets,JPEGImages,...}
    └── VOC2012/{Annotations,ImageSets,JPEGImages,...}
```

## 분포 (학습 / 평가 분할)
| 데이터셋 | 분할 | 샘플 수 | 비율 | 비고 |
|---------|------|--------|------|------|
| COCO 2017 | train | 118,287 | 96% | 공식 분할 그대로 |
| COCO 2017 | val | 5,000 | 4% | 공식 분할 — 본 표 baseline 산출 분할 |
| VOC 07+12 | trainval | 16,551 | 77% | VOC 07 trainval + VOC 12 trainval 합본 (**VOC 내부 표준 합본** — VOC 학습 시 단일 분할로 사용. COCO 와는 별개) |
| VOC 2007 | test | 4,952 | 23% | 공식 분할 — VOC mAP@0.5 산출 |

test 분할은 학습/검증에 **절대 사용 금지**. 누설 시 모든 결과 무효 (EVAL_PROTOCOL.md 의 원칙 인용).

## 레이블 / 어노테이션
- COCO: 80 클래스 (instance bbox), 어노테이션 도구 = Microsoft 공식 (`pycocotools` 로딩).
- VOC: 20 클래스 (object bbox), 어노테이션 도구 = Oxford VOC 공식 (XML 포맷, `xml.etree` 파싱).
- **합의 방식**: 공개 데이터셋 — 본 프로젝트가 라벨링 하지 않음.

## 결측 / 노이즈
- COCO: 일부 이미지에 어노테이션 없음 (background-only) — 학습에서 제외 (DiffusionDet 동치).
- VOC: `difficult=1` 박스는 평가에서 제외 (VOC 공식 컨벤션).

## 민감정보 (PII)
- 모두 공개 학술 데이터셋. PII 처리 정책 불필요.
- 모델 출력에 사람·차량 등 일반 클래스가 포함되지만, **운영 서비스로 배포하지 않으므로** PII 관점 영향 없음.

## 알려진 편향과 한계
- **COCO**: 미국 도시 중심 분포 — 일부 클래스(예: snowboard, surfboard) 가 서구 분포에 편향.
- **VOC**: 2007/2012 시점 자연사진 — 모바일 후방 카메라 / 드론 / 위성 분포에는 미적용.
- **공통**: 학습 시점 도메인 외 입력(의료/위성/저조도/X-ray)은 보장하지 않음 — PRD 의 "범위 외" 와 일치.
- **클래스 분포 불균형**: COCO 의 일부 long-tail 클래스(toaster 등)는 학습 샘플 부족 → AP 변동 큼. 슬라이스 분석은 EVAL_PROTOCOL.md.

## 관련 문서
- 평가 분할 사용: [EVAL_PROTOCOL.md](./EVAL_PROTOCOL.md)
- 데이터 의존성: [ARCHITECTURE.md#외부-의존](./ARCHITECTURE.md)
- 무엇을·왜: [PRD.md#데이터](./PRD.md)
