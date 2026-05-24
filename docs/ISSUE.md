# Issues — fm-det

> **이 문서가 답하는 질문**: 지금 *알려진 이슈는 무엇이며*, *과거에 해결된 이슈는 어떻게 발생하고 어떻게 풀렸는가*? Claude(또는 사람) 가 **모든 작업을 시작하기 전에 먼저 읽는** 문서 — 같은 이슈를 두 번 만나지 않기 위함.

## 사용 규칙
1. **작업 시작 전 본 문서를 먼저 읽는다.** ([../CLAUDE.md#이슈--현재-상태-관리](../CLAUDE.md) CRITICAL)
2. 새 이슈 발견 시 "진행 중" 섹션에 한 항목 append (아래 양식).
3. 이슈 해결 시 "해결됨" 섹션 맨 위로 이동, 해결책·재발 방지 한 줄 추가. **삭제 금지** — 미래의 같은 이슈에 컨텍스트 제공.
4. 한 줄 메모로 충분한 것은 [EXPERIMENTS.md](./EXPERIMENTS.md) "도구 학습 노트" 로. 본 문서는 *작업을 막거나·재발 가능성 있는* 이슈만.

## 양식
```markdown
### I-{NN}: {짧은 제목}
- **상태**: open / blocked / resolved
- **발견일**: YYYY-MM-DD
- **카테고리**: setup / data / model / training / eval / tooling / docs
- **증상**: {무엇이 잘못 동작하는가}
- **원인**: {추정 또는 확정 — 모르면 "조사 중"}
- **해결책 / 우회**: {적용된 방법 — open 이면 "검토 중"}
- **재발 방지**: {다음에 같은 상황을 만나면 무엇을 먼저 확인할지}
```

---

## 진행 중 (open / blocked)

### I-11: eval 에 NMS 미적용 — 한 객체에 중복 박스 → mAP 25% 천장 (실제 가중치는 70% 수준)
- **상태**: resolved (2026-05-24 — evals/voc.py 와 evals/coco.py 에 per-class `torchvision.ops.batched_nms` (iou=0.5) 추가)
- **발견일**: 2026-05-24
- **카테고리**: eval
- **증상**: I-10 (좌표계 fix) 적용 후에도 VOC mAP@0.5 가 0.2500 에서 정체. per-class AP 0.19~0.30 으로 균등하지만 천장이 낮음. inference overlay 보면 한 객체에 여러 head 의 refined box 가 같은 영역에 중복.
- **원인**: `evals/voc.py` / `evals/coco.py` 가 top-K 만 추출하고 NMS 를 적용하지 않음. 500 proposals × C class 중 score 상위 100 개를 그대로 prediction 으로 사용 → 한 객체에 N 개 박스가 있으면 가장 score 높은 1 개만 TP, 나머지 N-1 개는 모두 FP → mAP 하락. 원본 DiffusionDet inference 는 NMS `iou_thresh=0.5` 적용 (`DiffusionDet/.../diffusiondet.py` 의 `inference_*` 분기).
- **해결책 / 우회**: per-class NMS — `torchvision.ops.batched_nms(boxes, scores, class_idx, iou_threshold=0.5)`. [N, C] → [N*C] 펼침 → score thresh 먼저 (대부분 < 0.05 라 NMS 부담 경감) → batched_nms → top-100. 같은 ckpt (epoch 29 last.pt) VOC07 test full eval:
  - 좌표 fix 전: mAP@0.5 = 0.0000
  - 좌표 fix 후 (NMS 없음): mAP@0.5 = **0.2500**
  - 좌표 fix + NMS: mAP@0.5 = **0.7002** (+45 점, ~3 배)
- **재발 방지**: 새 eval pipeline 작성 시 (a) GT 좌표계 (b) NMS (c) score thresh (d) max_dets 네 가지 명시. evals/{voc,coco}.py 의 `nms_thresh` 파라미터 기본값 0.5 로 박힘. **mAP 가 짝수 비율 (예: 25%) 에서 멈춰서 더 안 오르면 첫 의심은 NMS 누락**.

### I-10: VOC eval 가 prediction 만 orig 좌표로 scale 하고 GT 는 cur 좌표로 둠 — mAP≈0 false-negative (실제 학습 정상)
- **상태**: resolved (2026-05-23 — evals/voc.py 의 prediction sx, sy scaling 제거, 둘 다 cur 좌표에서 비교)
- **발견일**: 2026-05-23
- **카테고리**: eval
- **증상**: VOC 학습이 epoch 0~25 내내 metric_primary=mAP@0.5≈0.0001 (per_class 거의 다 0). loss 는 38.95 → 3.5 로 정상 감소. eval 만 망가진 듯한 패턴 ("학습 안 되는 것 같다" 의문).
- **원인**: `datasets/transforms.py:46-51` 의 RandomResize 가 tgt["boxes"] 를 cur(resized) 좌표로 scale. `datasets/voc/dataset.py:87` 에서 tgt["size"] 는 cur 로 갱신되지만 tgt["orig_size"] 는 그대로 orig. `evals/voc.py:62-64` 가 prediction 만 `sx = orig_w/cur_w, sy = orig_h/cur_h` 로 scale 해서 orig 좌표로 보냄. 결과: pred(orig 좌표) vs GT(cur 좌표) box_iou → IoU 거의 0 → 모든 prediction FP → mAP≈0. 8 장 sanity check 결과 fix 후 mAP@0.5 = **0.0000 → 0.3251** (epoch 25 last.pt).
- **해결책 / 우회**: `evals/voc.py` 의 prediction sx, sy scaling 제거 (둘 다 cur 좌표에서 비교). mAP 의 정의는 ratio-based 라 좌표계 절대값 무관, 일치만 보장하면 됨. evals/coco.py 는 pycocotools 가 coco_gt(json, orig 좌표) 와 비교하므로 prediction → orig scale 이 *맞음* (다른 케이스). 둘을 헷갈리지 말 것.
- **재발 방지**:
  - 새 eval lib 추가 시 "GT 가 어디서 오는지" (target dict 인지 외부 ann json 인지) 와 "transforms 가 GT 를 건드리는지" 두 축 확인.
  - mAP=0 + loss 정상 감소 패턴이면 첫 의심은 **train/eval 좌표계 mismatch** (또는 class idx off-by-one). `phases/voc-repro-baseline/debug_eval_inference.py` 가 fix 전후 mAP 비교 + GT/pred overlay PNG 산출 — 다음에 같은 의심 시 첫 도구.
  - 학습 시작 시 epoch 1 후 mAP 가 정확히 0.0 (random init 보다도 낮음) 이면 즉시 eval 의심. random init 모델의 lower bound 가 0.001-0.003 정도이므로 epoch 1 에서 0.0000 은 의심 신호.

### I-09: 학습 python 프로세스가 부모 Claude 세션 종료 시 SIGHUP 으로 같이 사망 (no traceback)
- **상태**: resolved (2026-05-23 — 재시작 시 `nohup setsid ... < /dev/null &` 패턴으로 PPID=1 detach)
- **발견일**: 2026-05-23
- **카테고리**: tooling / training
- **증상**: P0 COCO (run_dir 0709, 0933) 와 VOC (run_dir 1233) 학습이 모두 **train.log 에 traceback 없이** 갑자기 멈춤. last 로그 라인이 정상 iter (예: VOC `iter 1800 loss=23.04 grad=67`) → 다음 raw 가 없음. metrics.csv 마지막 raw 와 같은 시각에 끊김. `ps -p <pid>` 찾을 수 없음 (`/proc/<pid>` 없음). GPU 점유 0%. 호스트 RAM·swap 여유 있음 (OOM 아님). dmesg 접근 불가 (컨테이너 내 root 아님).
- **원인**: 학습을 시작한 다른 Claude 세션 (PID 1537208 같은 별도 `claude` 프로세스) 이 종료될 때, 그 세션이 띄운 `python train.py` 가 child 였다면 **SIGHUP 전파로 같이 종료**. python 은 SIGHUP 의 default handler (terminate) 를 그대로 받음 → traceback 없이 silent exit. `nohup` 만으로는 부족할 수 있고 (setsid 가 없으면 같은 process group 에 머무름) — 진짜 detach 하려면 **`nohup setsid python ... < /dev/null > log 2>&1 &`** 로 새 session/pgrp + stdin 단절 까지 모두 필요.
- **해결책 / 우회**:
  - **재시작 패턴 (PPID=1 보장)**:
    ```bash
    nohup setsid env TORCH_HOME=/workspace/fm-det/.cache/torch \
      python train.py +experiment={tag} seed=42 \
      > phases/{tag}/train.log 2>&1 < /dev/null &
    # 진짜 python pid 는 setsid 후 fork 된 child — pgrep -af "train.py.*{tag}" 의 PPID=1 짜리.
    pgrep -af "train.py.*{tag}" | head -1 | awk '{print $1}' > phases/{tag}/train.pid
    ```
  - **검증**: `ps -p <pid> -o ppid,pgid,sid,stat,cmd` 에서 PPID=1, SID==PID, STAT 에 `s` (session leader) 포함이어야 함.
- **재발 방지**: 모든 장시간 학습은 **launcher script 의 한 줄로** 위 패턴 강제. `scripts/launch_train.sh` 같은 정본 헬퍼 신설을 다음 phase 에서 검토. 또한 `phases/{tag}/train.pid` 의 PID 가 PPID=1 짜리 python 인지 launch 직후 검증 한 줄 (`ps -o ppid= -p $(cat train.pid)` == 1) 추가 권장. `phases/{tag}/step0.md` 의 명령 예시도 `nohup setsid ... < /dev/null` 패턴으로 갱신 권장 (현재는 `nohup ... &` 만).

### I-08: execute.py 의 `_verify_success_metric` 가 `{run_dir}` 없는 code-only success_metric 을 강제 error 처리
- **상태**: resolved (2026-05-22 — code patch 적용)
- **발견일**: 2026-05-22
- **카테고리**: tooling
- **증상**: `kind: code` step (success_metric = `test -f X.py && python3 -c '...'` 류) 실행 시 Claude subprocess 가 정상 종료 + status=completed 마킹했어도, `_verify_success_metric` 이 `run_dirs` 가 비어 있다는 이유로 `False` 반환 → status 가 강제 error 로 전환. code-skeleton-loaders step 0 (transforms-common) 가 4회 재시도 모두 같은 사유로 실패.
- **원인**: `scripts/execute.py:290` 의 `if not run_dirs: return False, "..."` 가 success_metric 표현식에 `{run_dir}` 변수가 있든 없든 무조건 run_dirs 를 요구. 그러나 code-only step 의 success_metric 은 `test -f` / `python3 -c '...'` 처럼 파일 시스템·import 만 검사하고 `runs/{run_dir}/...` 산출물을 만들지 않음.
- **해결책 / 우회**: `_verify_success_metric` 를 *`{run_dir}` 가 expr 에 있을 때만* run_dirs 필수로 만들고, 없으면 expr 을 그대로 shell 실행. 또한 jq 미설치 skip 도 `"jq "` 가 expr 에 있을 때만 적용 (bash-only step 에서는 jq 없어도 OK). patch:
  ```python
  needs_run_dir = "{run_dir}" in expr
  if needs_run_dir and not run_dirs: return False, "..."
  if "jq " in expr and shutil.which("jq") is None: return True, "skip"
  cmd = expr.replace("{run_dir}", f"runs/{run_dirs[0]}") if run_dirs else expr
  ```
- **재발 방지**: 새 `kind` 추가 시 success_metric 표현이 `{run_dir}` 패턴을 사용하는지 (sanity/experiment/bench) 또는 file-import 패턴 (code) 인지 명확히. harness.md §5-3 의 "`{run_dir}` 변수가 들어가면" 조건문이 코드에 반영돼 있는지 검증. `_verify_success_metric` 수정 시 두 패턴 모두 unit test 권장.

### I-07: torch-cache named volume 영구화 깨짐 + root 소유 — `torchvision pretrained` 다운 실패
- **상태**: blocked (workaround 적용 가능 / 영구 fix 는 Dockerfile or entrypoint patch + rebuild 필요)
- **발견일**: 2026-05-22
- **카테고리**: setup
- **증상**: rebuild 후 `torchvision.models.resnet50(weights="DEFAULT")` 호출 시 `PermissionError: [Errno 13] Permission denied: '/home/docker_user/.cache/torch/hub'`. `ls -ld /home/docker_user/.cache/torch` → `drwxr-xr-x 2 root root`. docker_user (uid 1000) 가 mount-point 에 쓰기 불가. PENSIEVE 가 약속한 `resnet50-11ad3fa6.pth 97.8MB` 도 사라짐 (volume reset 추정).
- **원인**: docker-compose.yml 의 named volume `torch-cache:/home/docker_user/.cache/torch` 가 fresh 생성 시 Docker 가 mount-point 를 root 소유로 만듦. 이전에 한 번 chown 후 사용했어도 rebuild + volume reset (또는 `docker compose down -v`) 으로 다시 root 소유로 복귀. Dockerfile 에 `mkdir -p ~/.cache/torch && chown docker_user:docker_user ~/.cache/torch` 같은 pre-create 가 없음.
- **해결책 / 우회**:
  - **즉시 우회 (이 세션)**: `export TORCH_HOME=/workspace/fm-det/.cache/torch` (workspace bind-mount, docker_user 쓰기 가능). 워크스페이스에 `.cache/torch/hub/checkpoints/` 만들고 가중치 한 번 다운로드. `.gitignore` 에 `.cache/` 추가 권장.
  - **영구화 (rebuild 필요)**: env_docker/Dockerfile 에 `RUN mkdir -p /home/docker_user/.cache/{torch,huggingface,pip} && chown -R docker_user:docker_user /home/docker_user/.cache` (docker_user 전환 *전*) 추가. 또는 docker-entrypoint.sh 에 `sudo chown` 추가 (sudo 부재 시 root user 전환 후 drop). 또는 hf-cache / pip-cache 도 같은 패턴이므로 함께.
- **재발 방지**: named volume rebuild 시 mount-point 소유권 확인 — `docker compose -f env_docker/docker-compose.yml exec dev ls -ld /home/docker_user/.cache/{torch,pip,huggingface}` 가 모두 `docker_user docker_user` 이어야 함. Dockerfile 에 pre-create + chown 패턴 박힘으로 첫 빌드부터 보장.

### I-04: DiffusionDet 본 repo 재현치(AP 46.2) 미검증 + 메커니즘 진단 미수행
- **상태**: open
- **발견일**: 2026-05-21
- **카테고리**: training
- **증상**: 본 프로젝트의 모든 ablation 결과가 "AP 46.2 ± 0.5" baseline 위에 서야 의미를 가진다 ([PRD.md#성공-기준](./PRD.md)). 또한 fm-det 의 "내공 형성" 목표는 P0a 진단 실험(최소 5행)이 채워져야 시작 — 아직 P0/P0a 학습 미실행.
- **원인**: 학습 시작 전 (현재는 docs/구조 설계 단계).
- **해결책 / 우회**: `/harness` 로 P0 (`{ds}-repro-*`) phase 설계 → 학습 + AP 측정 → ±0.5 이내면 P0a (`{ds}-diag-*`) 진단 5행 실행 → P1 FM 전환. 미달이면 후속 phase 보류.
- **재발 방지**: P0 baseline 매칭 + P0a 진단 5행 채우기 전에 P1/P2 학습 시작 금지. [EXPERIMENTS.md#단계적-FM-전환-로드맵](./EXPERIMENTS.md) 의 P0 → P0a → P1 순서 준수.

---

## 해결됨 (resolved — 미래의 같은 이슈에 참고)

### R-08: PyTorch sm_120 (Blackwell) 미지원 — 2.7.1+cu128 base 로 해결 (구 I-06)
- **해결일**: 2026-05-22
- **카테고리**: training
- **발생 경위**: 2026-05-21 발견. RTX PRO 6000 Blackwell (sm_120) 에서 forward 시 `CUDA error: no kernel image is available`. PyTorch 2.5.1 / 2.6.0 stable wheel 의 `torch.cuda.get_arch_list()` 에 sm_120 없음.
- **해결책**: env_docker/Dockerfile base 를 **`pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel`** 로 patch (PyTorch 2.7.0 stable 이 sm_120 첫 공식 지원, 2.7.1 은 patch). 1차 patch (2.6.0+cu124) 가 wheel arch_list 에 sm_120 미포함이라 실패한 후 2차 patch 로 영구화. 2026-05-22 호스트 rebuild 검증: `torch.__version__ == '2.7.1+cu128'`, `arch_list = [..., 'sm_120', 'compute_120']`, `cuda.get_device_name(0) == 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'`, Linear(10,10).cuda() forward PASS.
- **재발 방지**: 새 GPU 도입 시 (a) `torch.cuda.get_device_capability()` (b) `torch.cuda.get_arch_list()` 둘 다 확인. 교집합 비면 부적합. PyTorch release note 의 "supported architectures" 절을 base 이미지 선정 전 확인 — 버전 ≥ X 만으로 추정 금지 (1차 patch 실패 사유). Dockerfile base 주석에 sm_120 공식 지원 = 2.7.0 stable 첫 release 박힘.

### R-07: `jq` / `unzip` 미설치 — rebuild 로 영구 설치됨 (구 I-05)
- **해결일**: 2026-05-22
- **카테고리**: tooling
- **발생 경위**: 2026-05-21 발견. dev 컨테이너에 `jq` 부재로 execute.py 의 `_verify_success_metric` skip. `unzip` 부재로 voc archive 압축 해제 시 python zipfile 우회 필요.
- **해결책**: env_docker/Dockerfile 의 apt-get install 라인에 `jq unzip` 추가. 2026-05-22 호스트 rebuild 후 `jq --version → jq-1.6`, `unzip -v → UnZip 6.00` 확인.
- **재발 방지**: dev 컨테이너 새 빌드 시 entrypoint 헬스체크 한 줄 (`jq --version && unzip -v`) 권장. harness 정본 §5-3 "jq 설치 의무" 가 Dockerfile 주석에 박힘 (R-07 시점에 미러됨).

### R-06: `.gitignore` 가 ai-ml 정책을 반영하지 않음
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: 초기 `.gitignore` 는 node_modules / phases 산출물 / .env 만 포함. `data/`, `runs/`, `wandb/`, `outputs/`, `.hydra/`, `*.pt/.pth/.ckpt/.safetensors`, `__pycache__/`, `*.egg-info` 등 ai-ml 산출물이 추적 대상이라 데이터 다운(`data/coco/`)·실험 산출물(`runs/data-sanity-*`) 직후 git status 가 어수선.
- **해결책**: `.gitignore` 에 ai-ml 섹션 (`data/`, `runs/`, `wandb/`, `outputs/`, `.hydra/`, `*.pt/.pth/.ckpt/.safetensors`, `__pycache__/`, `*.py[cod]`, `*.egg-info/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `build/`, `dist/`, `.DS_Store`, `.vscode/`, `.idea/`) 추가. `git check-ignore -v` 로 `data/coco/val2017`, `runs/data-sanity-download-*/manifest.json` 매치 확인.
- **재발 방지**: 새 도구 추가 시 (예: `mlflow/`, `dvc/`) `.gitignore` 도 함께 갱신. PreToolUse hook (가중치·data/ 강제 추가 차단) 과 이중 안전망.

### R-05: `pyproject.toml` 미도입 — 루트 패키지 import 가 PYTHONPATH 의존
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: `src/` 폐지 + 루트 평탄 배치 (ADR-005) 후 패키지 메타데이터 미작성. `from datasets import ...` 가 cwd 의존 → 다른 디렉터리에서 호출 시 `ModuleNotFoundError`.
- **해결책**: `pyproject.toml` 신설 — `[build-system]` setuptools 68+, `[project]` name=fm-det version=0.0.1 requires-python>=3.11, `[tool.setuptools.packages.find]` 의 include 에 `datasets*, models*, losses*, evals*, utils*` / exclude 에 `DiffusionDet*, data*, runs*, wandb*, outputs*, env_docker*, scripts*, phases*, docs*, tests*, .claude*`. `pip install -e .` 로 editable install 가능. (현재 모듈 폴더는 비어 있어 install 자체는 trivial — P0 baseline phase 시 datasets/coco.py 등 추가 시 실제 동작 확인 필요.)
- **재발 방지**: 새 루트 패키지 추가 시 (예: `transforms/`) `pyproject.toml` 의 packages.find.include 에 함께 추가. 폴더 신설 ≠ 자동 패키지화 — `__init__.py` + pyproject 의 include 양쪽 필요.

### R-04: Claude Code statusLine 미표시 — `settings.json` 의 호스트 경로 하드코딩
- **해결일**: 2026-05-21
- **카테고리**: tooling
- **발생 경위**: `~/.claude/settings.json` 의 `statusLine.command` 가 `bash /home/sungmin-cho/.claude/statusline-command.sh` 로 호스트 사용자 경로 하드코딩. 컨테이너 내부 사용자(`docker_user`) 에는 그 경로가 없어 statusLine 실행 자체가 실패 → 화면에 아무것도 안 보임. 또한 statusline 스크립트의 `${USER}` 가 컨테이너 실행 컨텍스트에서 비어 나와 `@hostname` 으로 표시되는 부수 이슈.
- **해결책**: (1) `settings.json` 의 command 를 `bash $HOME/.claude/statusline-command.sh` 로 변경 — 호스트/컨테이너 양쪽에서 동작. (2) statusline-command.sh 의 `"${USER}"` → `"${USER:-$(whoami)}"` 로 fallback 추가.
- **재발 방지**: `~/.claude/settings.json` 의 command 경로에 절대 사용자 홈 경로 하드코딩 금지 — 항상 `$HOME` 또는 `~` 사용. 호스트/컨테이너 동기화 설정 파일은 사용자별 차이를 추상화해야 함.

### R-03: `env_docker/` 미생성 — `/docker-init` 호출로 해소
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: docs 채우기 단계까지 컨테이너 환경 없이 호스트 Python 으로 회귀를 돌렸음. 학습은 GPU 의존이라 호스트에서 실행 불가.
- **해결책**: `/docker-init` 호출 → `env_docker/{Dockerfile, docker-compose.yml, docker-entrypoint.sh, .dockerignore}` + `Makefile` + `.env.example` + `requirements.txt` 생성. 베이스 이미지 = `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel`, GPU runtime(`runtime: nvidia` + `deploy.resources.reservations.devices`), `shm_size: 8gb`, TB 포트 `6007:6006`, `~/.claude` 마운트로 인계.
- **재발 방지**: 컨테이너 의존성(PyTorch / CUDA / Hydra / W&B / pycocotools) 변경 시 `env_docker/Dockerfile` + `requirements.txt` 도 함께 갱신. 호스트에서 직접 `python train.py` 실행 금지 — 항상 `make shell` 후 컨테이너 안에서. 호스트 UID/GID 미스매치 권한 에러는 `.env` 의 `HOST_UID`/`HOST_GID` (`id -u`/`id -g` 결과) 로 보정.

### R-01: bootstrap 시점 `.git` 디렉터리 부재
- **해결일**: 2026-05-21
- **카테고리**: setup
- **발생 경위**: bootstrap §1 인프라 점검 후 git status 가 `fatal: 깃 저장소가 아닙니다` — README Quick Start 0단계의 `git init` 누락. 시스템 컨텍스트의 "Is a git repository: true" 와 어긋났음.
- **해결책**: `git init` 실행 → "기존 깃 저장소를 다시 초기화" 메시지 → 정상화.
- **재발 방지**: 새 프로젝트 클론 직후 `git init` 을 확인. bootstrap 의 §3-B-1 git 기반 dirty 점검은 git 저장소가 있어야 의미 — 없으면 사용자 콘텐츠 의심 분기로 보수적으로 진행.

### R-02: bootstrap.md 기댓값 (`91 passed, 2 skipped`) vs 실측 (`121 passed`)
- **해결일**: 2026-05-21
- **카테고리**: tooling
- **발생 경위**: ai-ml 박물관 §3-B-4 회귀 검증에서 bootstrap.md 의 기댓값보다 실측 테스트 수가 30개 많음.
- **해결책**: 박물관 정본(`templates/ai-ml/scripts/test_execute.py`) 이 그 사이 테스트가 추가됐을 뿐 — 실패가 없으므로 정상. bootstrap.md 의 기댓값 줄이 outdated 일 뿐 동작은 OK.
- **재발 방지**: 회귀 검증 판단 기준은 "실패 0 / skip 0~3" 가 본질. 기댓값 정확 매칭이 아님.
