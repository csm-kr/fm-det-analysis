# Weights & Biases Guide (첫 도입자용 학습 노트)

> **이 문서가 답하는 질문**: W&B 가 *무엇이고* TensorBoard 와 어떻게 *공존*하며, fm-det 에서 *어떻게 설정·로깅·sweep* 하는가? 실제 운영 컨벤션은 [EXPERIMENTS.md](./EXPERIMENTS.md), 도입 결정 근거는 [ADR.md#ADR-003](./ADR.md).

## 한 줄 요약
W&B = **클라우드 기반 실험 추적 도구**. TensorBoard 의 스칼라/이미지/하이퍼파라미터 로깅에 더해 **여러 실험 한 화면 비교 / sweep 자동 실행 / artifact 버저닝** 까지 제공. fm-det 은 TensorBoard (기준, 오프라인) + W&B (병행, 비교 편의) 를 함께 쓴다.

---

## 왜 W&B 인가 (TB 와 같이 쓰는 이유)
- TB 단독은 ablation 다수 비교 불편 (탭 전환, run 이름 필터링 어려움).
- W&B 는 같은 프로젝트 안 모든 run 을 한 표 / 한 그래프로 비교, 정렬, 필터, group-by 가능.
- sweep: yaml 한 장으로 hyperparameter 조합 자동 탐색.
- 단점: 외부 SaaS (오프라인 환경 제약) → fm-det 은 **offline 모드 우선** + 익숙해진 후 sync.

---

## 1) 계정 / API 키 / 첫 설치

### 가입 & API key
1. https://wandb.ai/ 에서 계정 만들기 (개인 무료 계정으로 충분).
2. https://wandb.ai/authorize 에서 API key 복사.
3. 컨테이너 환경변수로 주입 — **레포에 절대 커밋 금지**:
   ```bash
   # ~/.bashrc (호스트) 또는 env_docker/.env (gitignore 된 위치)
   export WANDB_API_KEY=your_key_here
   ```
4. `env_docker/docker-compose.yml` 의 `environment:` 또는 `env_file:` 로 컨테이너에 전달.

### 패키지 설치
```bash
pip install wandb
```

### 첫 로그인 (한 번만)
```bash
wandb login           # 환경변수에 키가 있으면 자동, 없으면 prompt
```

---

## 2) 핵심 개념 4개

### Project / Run / Group / Tags
- **Project**: 실험 모음 (fm-det 은 단일 project = `fm-det`).
- **Run**: 학습 1회 (= `runs/{ts}-{tag}/`).
- **Group**: 여러 run 묶기 (예: `coco-fm1` group 에 시드 3개 run).
- **Tags**: 검색 / 필터용 라벨 (예: `coco`, `fm1`, `sampler-cfm`).

### Online / Offline 모드
| 모드 | 동작 | 추천 시점 |
|------|------|----------|
| `online` | 학습 중 실시간 SaaS 업로드 | 익숙해진 후 / 안정 환경 |
| `offline` | 로컬 `wandb/` 폴더에만 기록, 나중에 `wandb sync` 로 업로드 | **fm-det 초기 default** |
| `disabled` | 아무것도 안 함 (디버깅 시) | 잠시 끄고 싶을 때 |

설정:
```python
import wandb
wandb.init(project="fm-det", mode="offline", dir=cfg.run_dir)  # runs/{id}/wandb/ 에 기록
```

또는 환경변수 `WANDB_MODE=offline`.

### Artifact (체크포인트 버저닝 — 선택, 후반에 도입)
- `wandb.Artifact("fm-det-checkpoints", type="model")` 로 `best.pt` 를 버전 관리.
- 단점: 용량 크고 SaaS quota 소모 → fm-det 은 **로컬 checkpoints 우선**, artifact 는 결론 후보 모델만.

### Sweep (hyperparameter 자동 탐색)
- yaml 한 장에 탐색 공간 정의 → W&B 가 agent 띄워 자동 실행.
- fm-det 은 처음에 sweep 안 씀 (Hydra `--multirun` 으로 충분). 익숙해진 후 도입.

---

## 3) fm-det 통합 패턴 (실전)

### Hydra + W&B + TB 동시 로깅
```python
# utils/logging.py (도입 phase 의 산출물 — 참고)
import torch.utils.tensorboard as tb
import wandb

class Logger:
    def __init__(self, cfg):
        self.tb = tb.SummaryWriter(log_dir=f"{cfg.run_dir}/tb_logs")
        wandb.init(
            project="fm-det",
            name=cfg.experiment.tag,           # runs/ 의 tag 와 동일
            group=cfg.experiment.phase,        # P0, P1, P2 ... 로 grouping
            tags=[cfg.data.name, cfg.experiment.phase],  # ["coco", "P1"] 등
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=cfg.run_dir,                   # runs/{id}/wandb/
            mode=cfg.wandb.mode,               # "offline" default
        )

    def log_scalar(self, key: str, value: float, step: int):
        self.tb.add_scalar(key, value, step)
        wandb.log({key: value}, step=step)
```

### 학습 중 로깅
- 매 step: `train/loss`, `lr`, `samples/sec`.
- 매 epoch 끝: `val/AP`, `val/AP50`, `val/AP75`, `val/loss`, `GPU memory`.
- 결론 후보 run 만 이미지 시각화 (`wandb.log({"pred_examples": wandb.Image(...)})`) — 용량 절약.

### Offline → Online sync
```bash
# 학습 끝난 뒤 (또는 정기적으로)
wandb sync runs/{ts}-{tag}/wandb/offline-run-*
```

---

## 4) 자주 만나는 함정 (W&B 첫 도입 시)

| 증상 | 원인 / 해결 |
|------|-----------|
| `wandb.errors.UsageError: api_key not configured` | `WANDB_API_KEY` 환경변수 누락 또는 `wandb login` 안 함. |
| 학습 중 SaaS 업로드 지연 → step 느려짐 | `online` 모드의 네트워크 병목. `offline` 모드로 변경 후 학습 종료 후 sync. |
| `wandb.init` 두 번 호출 → 이전 run 끝남 | DDP 환경에서 rank 0 만 init 해야 함 — `if rank == 0: wandb.init(...)`. |
| run 이름이 자동 생성 ("dazzling-fish-12") | `wandb.init(name=...)` 으로 명시. fm-det 은 `cfg.experiment.tag` 사용. |
| TB 와 W&B 의 metric 이 다르게 보임 | `wandb.log({"x": v}, step=i)` 와 `tb.add_scalar("x", v, i)` 의 step 인자 일치 확인. |
| `Run history` 가 비어 있음 | `wandb.log` 호출 안 함. `wandb.config` 에 hyperparameter 만 넣고 metric 로깅 누락 사례. |
| sweep agent 가 즉시 종료 | sweep yaml 의 `program: train.py` 에서 진입점 경로/CLI 인자 형식 확인. Hydra 와 함께 쓰려면 `python train.py +experiment=${experiment}` 식. |
| GPU OOM 이 아닌데 SaaS 응답 timeout | `WANDB__SERVICE_WAIT=300` 환경변수로 timeout 늘림. 또는 `offline`. |

---

## 5) TB 가 따로 필요한 이유 (오해 방지)
W&B 가 있는데 왜 TB 도 쓰나?
- **외부 SaaS 장애 / 오프라인 환경 보험** — TB 는 항상 동작.
- **재현성 보장** — 외부 서비스가 사라져도 `runs/{id}/tb_logs/` 만 있으면 재시각화 가능.
- **이미지/spectrogram 등 무거운 로깅** — W&B 무료 quota 소모를 피하기 위해 TB 만 사용.

---

## 학습 자료 (외부)
- 공식 quickstart: https://docs.wandb.ai/quickstart
- Sweep tutorial: https://docs.wandb.ai/guides/sweeps
- PyTorch 통합: https://docs.wandb.ai/guides/integrations/pytorch
- Offline mode: https://docs.wandb.ai/guides/technical-faq/general#how-do-i-run-wandb-offline

## 학습 노트 (개인 누적)
> W&B 사용 중 깊이 파게 된 항목을 누적. 한 줄 짧은 메모는 [EXPERIMENTS.md](./EXPERIMENTS.md) 에.
- `{YYYY-MM-DD}` — `{주제}` — `{이해한 내용 / 참고 링크}`
