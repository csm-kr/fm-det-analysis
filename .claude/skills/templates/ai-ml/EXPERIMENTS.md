# 실험 운영

## 디렉터리 컨벤션
모든 학습 실행은 다음 구조의 결과를 남긴다:

```
runs/{YYYYMMDD-HHmm}-{tag}/
├── config.yaml         # 실행 시점의 하이퍼파라미터 스냅샷 (자동 복사)
├── git_rev.txt         # 코드 커밋 해시
├── seed.txt            # 시드값
├── metrics.csv         # epoch별 메트릭 시계열
├── checkpoints/
│   ├── best.pt
│   └── last.pt
└── logs/               # tensorboard event / 텍스트 로그
```

`tag` 는 사람이 읽는 짧은 라벨 (예: `baseline`, `lr1e-3`, `aug-strong`).

## 시드 정책
- 모든 학습 스크립트는 `--seed` 인자를 받는다. 기본값: {예: 42}
- 다음을 모두 고정: `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`
- DataLoader 의 `worker_init_fn` 도 시드 고정
- 비교 실험은 시드 {예: 3개} 평균/표준편차로 보고

## 하이퍼파라미터 스냅샷
- 학습 시작 직후 `configs/{name}.yaml` 을 `runs/{id}/config.yaml` 로 그대로 복사
- 같은 시점에 `git rev-parse HEAD` 결과를 `git_rev.txt` 에 기록
- 코드와 설정 둘 다 변경 추적 가능해야 한다 — "이 결과를 어떻게 만들었는가" 답변 가능 의무

## 실험 추적 도구
- 도구: {예: TensorBoard / W&B / MLflow / 자체 CSV}
- 기록 항목: {예: train/val loss, train/val 메트릭, 학습률, GPU 메모리}
- 외부 SaaS 사용 시: {API 키 환경변수 이름, 프로젝트 이름}

## 데이터 버전
- 데이터셋 변경 시 `data/CHANGELOG.md` 에 한 줄 의무
- 형식: `YYYY-MM-DD - {변경 한 줄} - {새 샘플 수}`
- 학습 스크립트는 데이터셋 해시(또는 manifest) 도 `runs/{id}/` 에 기록
