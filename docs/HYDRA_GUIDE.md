# Hydra Guide (첫 도입자용 학습 노트)

> **이 문서가 답하는 질문**: Hydra 가 *무엇이고* 왜 쓰며, fm-det 에서 *어떻게 ablation 한 행을 추가*하는가? 실제 운영 컨벤션은 [EXPERIMENTS.md](./EXPERIMENTS.md), 도입 결정 근거는 [ADR.md#ADR-005](./ADR.md).

## 한 줄 요약
Hydra = **YAML config 들을 그룹으로 나누고, defaults 리스트와 CLI override 로 조합/덮어쓰는 도구**. ablation 한 행을 한 yaml 로 표현 → `python train.py +experiment={tag}` 한 줄로 실행.

---

## 왜 Hydra 인가 (fm-det 에서)
- ablation 표 한 행 ↔ `configs/experiment/{tag}.yaml` 1 파일 ↔ `runs/{ts}-{tag}/` 1 디렉터리 의 1:1 매핑이 자연스러움.
- CLI 에서 `seed=17`, `train.lr=1e-4` 처럼 즉석 override 가능 — 코드 수정 없이 ablation 변종 추가.
- output 디렉터리·실행 시점 config 스냅샷·sweep (여러 ablation 자동 실행) 까지 표준 제공.

대안과 비교 (왜 다른 걸 안 썼는지)는 [ADR.md#ADR-005](./ADR.md) 참고.

---

## 핵심 개념 5개

### 1) Config Group
관련 yaml 들을 폴더로 묶은 것. fm-det 의 그룹:
```
configs/model/      → ResNet50, Swin-T 등 backbone 변종
configs/data/       → coco.yaml, voc.yaml
configs/train/      → baseline.yaml (lr 2.5e-5, AdamW, ...)
configs/loss/       → diffusion.yaml, flow_matching.yaml
configs/experiment/ → ablation 한 행 (이 그룹이 다른 그룹들을 선택)
```

### 2) Defaults 리스트
yaml 파일 안에서 다른 group 의 어떤 변종을 쓸지 선언:
```yaml
# configs/train.yaml (진입점 base)
defaults:
  - model: resnet50      # configs/model/resnet50.yaml 을 선택
  - data: coco           # configs/data/coco.yaml 을 선택
  - train: baseline      # configs/train/baseline.yaml 을 선택
  - loss: diffusion      # configs/loss/diffusion.yaml 을 선택
  - _self_               # 본 파일의 추가 키들

seed: ???                # 필수 (CRITICAL — 없으면 에러)
hydra:
  run:
    dir: runs/${now:%Y%m%d-%H%M}-${experiment.tag}
```

### 3) Experiment 파일 (ablation 한 행)
defaults override + 한 변수 변경. 예시 — sampler 를 FM 으로 바꾼 ablation:
```yaml
# configs/experiment/coco-fm1-sampler-cfm.yaml
# @package _global_
defaults:
  - override /loss: flow_matching   # diffusion → flow_matching 으로 1 변수만 변경
  - _self_

experiment:
  tag: coco-fm1-sampler-cfm         # runs/{ts}-{tag}/ 의 tag
  phase: P1
  changed_variable: "loss: diffusion → flow_matching"   # ablation 표 컬럼
```

> `# @package _global_` 헤더는 experiment yaml 의 키를 최상위에 병합한다는 의미.

### 4) CLI Override
실행 시점에 키 즉석 변경:
```bash
python train.py +experiment=coco-fm1-sampler-cfm seed=42
python train.py +experiment=coco-fm1-sampler-cfm seed=17 train.lr=1e-4   # 시드+lr 동시 override
```

`+experiment=...` 의 `+` 는 "기존에 없던 키 추가" 의미 (experiment 그룹은 defaults 에 없으므로 `+` 필요).

### 5) Sweep (선택 — 나중에 익숙해진 후)
여러 ablation 을 한 번에 실행 (multirun):
```bash
python train.py --multirun +experiment=coco-fm1-sampler-cfm,coco-fm1-sampler-rectified seed=42,17
# → 2 experiment × 2 seed = 4 run 자동 실행
```

---

## ablation 한 행 추가 절차 (실전 워크플로우)

1. **base group 변종이 필요한가?** — 새 모듈 변종(예: encoder 를 Swin-T 로 교체)이면 `configs/model/swin_t.yaml` 신설.
2. **`configs/experiment/{ds}-{phase}-{module}-{변경}.yaml` 작성** — defaults override 한 줄 + `experiment.tag` + `experiment.changed_variable`.
3. **실행** — `python train.py +experiment={tag} seed=42`.
4. **결과 확인** — `runs/{ts}-{tag}/config.yaml` (Hydra 자동 스냅샷) + `metrics.csv` + `tb_logs/` + `wandb/`.
5. **ablation 표 갱신** — `runs/{ts}-{tag}/eval.json` → [EXPERIMENTS.md](./EXPERIMENTS.md) 의 ablation 표 한 줄 append (`scripts/append_experiments.py` 가 자동, 도입 예정).

---

## 자주 만나는 함정 (Hydra 첫 도입 시)

| 증상 | 원인 / 해결 |
|------|-----------|
| `+experiment=foo` 가 에러 — "Could not override 'experiment'" | `experiment` 가 defaults 에 없는 그룹이라 `+` 필요. `experiment=foo` (X) → `+experiment=foo` (O). |
| override 가 무시됨 — `seed=17` 했는데 42 로 학습됨 | yaml 안에서 `seed: 42` 가 하드코딩됐을 가능성. `seed: ???` (placeholder) 로 두면 CLI 또는 experiment yaml 에서 강제 override. |
| `defaults` 의 다른 그룹을 experiment yaml 에서 override 가 안 됨 | `defaults:` 안에서는 `- override /group: variant` 형식이어야 함 (slash 와 override 키워드 필수). |
| 모든 run 이 `outputs/{date}/{time}/` 에 저장됨 | `hydra.run.dir` override 안 함. `configs/train.yaml` 의 `hydra.run.dir` 을 `runs/{...}` 로 설정. |
| `${now:%Y%m%d}` 가 그대로 문자열로 들어감 | OmegaConf resolver 가 `${now:...}` 만 지원. 다른 변수 보간은 `${experiment.tag}` 형식 (점 표기). |
| Tab 들여쓰기 사용 → yaml 파싱 에러 | YAML 은 space 만. `tab` 대신 space 2~4 칸. |
| `# @package _global_` 빠짐 → experiment 키가 `experiment.` 네임스페이스 안에만 들어감 | experiment yaml 첫 줄에 반드시 추가. |

---

## 최소 구현 (도입 phase 의 산출물 — 참고)
```python
# train.py
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.seed is not None, "CRITICAL: --seed (또는 seed=...) 필수"
    print(OmegaConf.to_yaml(cfg))   # 스냅샷 확인용
    # ... 시드 고정, 데이터 로더, 모델, 학습 루프

if __name__ == "__main__":
    main()
```

---

## 학습 자료 (외부)
- 공식 튜토리얼: https://hydra.cc/docs/tutorials/intro/
- Defaults 리스트 심화: https://hydra.cc/docs/advanced/defaults_list/
- Override grammar: https://hydra.cc/docs/advanced/override_grammar/basic/

## 학습 노트 (개인 누적 — EXPERIMENTS.md "도구 학습 노트" 와 별개로 깊은 항목)
> Hydra 사용 중 깊이 파게 된 항목을 누적. 한 줄 짧은 메모는 [EXPERIMENTS.md](./EXPERIMENTS.md) 에.
- `{YYYY-MM-DD}` — `{주제}` — `{이해한 내용 / 참고 링크}`
