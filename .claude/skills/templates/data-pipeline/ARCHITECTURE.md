# 아키텍처

> **이 문서가 답하는 질문**: 소스→변환→싱크의 *토폴로지·실패 분기·오케스트레이션*은 어떻게 구성되는가? 입출력 계약은 [DATA_CONTRACTS.md](./DATA_CONTRACTS.md), 장애 대응은 [RUNBOOK.md](./RUNBOOK.md), *무엇을·왜* 는 [PRD.md](./PRD.md).

## 시스템 구조
```
[소스]                  [파이프라인]                       [싱크]
┌──────────────┐
│운영 PostgreSQL│ ─┐
└──────────────┘   │   ┌─────────┐  ┌──────────┐  ┌──────┐    ┌─────────────┐
┌──────────────┐   ├─▶ │ extract │─▶│ transform│─▶│ load │ ─▶ │ BigQuery    │
│S3 로그        │ ─┤   └─────────┘  │ 검증·정제│  └──────┘    │ Snowflake   │
└──────────────┘   │                │ 조인·집계│        │     └─────────────┘
┌──────────────┐   │                └────┬─────┘        ▼
│Kafka 토픽     │ ─┘                     │ 실패     ┌─────────────┐
└──────────────┘                         ▼          │ S3 Parquet  │
                                    ┌─────────┐     └─────────────┘
                                    │ DLQ +   │
                                    │ 알림    │
                                    └─────────┘
                ▲
                │ 트리거
        ┌───────────────┐
        │ 오케스트레이터│  {Airflow / Prefect / Dagster / cron}
        └───────────────┘
```
> 입출력 스키마 합의는 별도 문서: [DATA_CONTRACTS.md](./DATA_CONTRACTS.md). 장애 대응: [RUNBOOK.md](./RUNBOOK.md).

## 디렉터리 구조
```
src/
├── extract/           # 소스에서 가져오기
├── transform/         # 정제, 조인, 집계
├── load/              # 싱크에 적재
├── jobs/              # DAG / 워크플로우 정의
├── schemas/           # 입출력 스키마 — 상세는 DATA_CONTRACTS.md
└── utils/             # 공통 헬퍼

dags/                  # 오케스트레이터 DAG (별도 디렉터리)
configs/               # 환경별 연결 정보
```

## 처리 흐름
1. 스케줄 트리거 (또는 이벤트)
2. extract: 소스 읽기 (증분 / 전체) — 마지막 watermark 기준
3. validate: 스키마·범위 검증 ([DATA_CONTRACTS.md](./DATA_CONTRACTS.md) 의 계약 위반 시 격리)
   - 통과 → 다음
   - 실패 → DLQ + 알림 → 종료
4. transform: 정제·조인·집계 — 단계별로 idempotent
5. load: 싱크에 적재 (upsert / merge)
6. verify: 건수·체크섬 검증
   - 통과 → 완료, watermark 갱신
   - 실패 → 롤백 + 알림 → [RUNBOOK.md](./RUNBOOK.md) 절차

## 오케스트레이션
- 도구: {예: Airflow / Prefect / Dagster / cron}
- 스케줄: {예: 일 02:00 KST / 매 5분}
- 의존성: {DAG 트리/엣지 핵심만}

## 데이터 모델
- 입력 스키마: {Avro / JSON Schema / Protobuf / 자유 형식}
- 출력 스키마: {파티션 키, 컬럼, 데이터 타입}
- 파티셔닝: {예: 일 단위 / 이벤트 시간}
- 상세 합의 형식·버전 관리: [DATA_CONTRACTS.md](./DATA_CONTRACTS.md)

## 외부 의존
| 종류 | 대상 | 용도 |
|------|------|------|
| 소스 | {예: PostgreSQL / Kafka} | {용도} |
| 싱크 | {예: BigQuery / S3} | {용도} |
| 모니터링 | {예: Datadog / Grafana} | {용도} |

## Hook 정책
- 자동 포맷터: {예: ruff format / sqlfluff / 없음}
- 추가 차단 패턴: {예: 운영 DB 자격증명 커밋 차단 / 없음}
- Stop hook 추가 검증: {예: pytest tests/transform 단위 / dbt build --select state:modified / 없음}

## 관련 문서
- 무엇을·왜: [PRD.md](./PRD.md)
- 결정 근거: [ADR.md](./ADR.md)
- 입출력 계약: [DATA_CONTRACTS.md](./DATA_CONTRACTS.md)
- 장애 대응: [RUNBOOK.md](./RUNBOOK.md)
