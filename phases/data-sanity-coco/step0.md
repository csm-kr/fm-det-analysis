# Step 0: download-coco-val

## 읽어야 할 파일

- `/CLAUDE.md` (CRITICAL — detectron2 금지 / 시드 의무 / 평가 프로토콜 변경 금지)
- `/docs/DATA_CARD.md` (다운로드 대상 URL / 디렉터리 레이아웃 / 분포 표)
- `/docs/EVAL_PROTOCOL.md` (test split 사용 금지 룰)

## 작업

COCO 2017 의 **val + annotations** 만 받아 무결성 검증한다. train2017 (18GB) 은 본 step 범위 밖.

### 1. 진입점

`datasets/coco/download.py` 신설 (루트 평탄). DATA_CARD 의 `scripts/data_download.sh` 는 outdated — `scripts/` 는 하네스 정본 전용이라 데이터 다운 진입점은 루트로. 본 step 마무리 시 DATA_CARD 의 해당 줄 갱신.

시그니처:
```python
def download(target: str, split: str, data_root: Path, run_dir: Path, seed: int) -> dict:
    """COCO 다운로드 + 압축 해제 + 무결성 검증 + manifest 작성.
    반환: manifest dict (그대로 run_dir/manifest.json 에 dump).
    """
```

CLI:
```bash
python -m datasets.coco.download --target coco --split val --data-root data --seed 42
# → data/coco/val2017/{*.jpg}, data/coco/annotations/{*.json}
# → runs/data-sanity-download-{YYYYMMDD-HHmm}/manifest.json
```

### 2. 다운로드 대상

| 파일 | URL | 크기 |
|------|-----|------|
| val2017.zip | http://images.cocodataset.org/zips/val2017.zip | ~1.0 GB |
| annotations_trainval2017.zip | http://images.cocodataset.org/annotations/annotations_trainval2017.zip | ~250 MB |

저장 경로:
- 다운로드 zip 임시: `data/coco/_zips/{val2017,annotations_trainval2017}.zip`
- 압축 해제 후: `data/coco/val2017/`, `data/coco/annotations/`
- 압축 해제 검증 통과 시 zip 은 보존 (재다운 방지) 또는 삭제 — 본 step 은 **보존** (디스크 여유 충분 + 재현성).

### 3. 다운로드 전략

- `wget -c` (resume) + `--tries=3 --timeout=60`
- 이미 받았으면 skip (파일 크기 일치 시).
- 압축 해제는 `unzip -q -d {dest}` — 이미 풀려 있고 파일 개수 일치 시 skip.

### 4. 무결성 검증

- 각 zip: `unzip -t` 통과 (CRC OK)
- 각 zip: SHA256 계산 → manifest 에 기록 (공식 체크섬 미공개라 첫 다운 시 자체 기록 → 향후 재다운 시 비교 자료)
- `data/coco/val2017/*.jpg` 개수 == 5000
- `data/coco/annotations/*.json` 개수 >= 4 (instances_train2017 / instances_val2017 / captions_*/ keypoints_* 6 개 중 detection 만 필요한 instances_*.json 2 개는 반드시 포함)
- annotations 디렉터리 안에 `instances_val2017.json` 존재 (pycocotools 로딩 직전 확인)

### 5. manifest.json 스키마

```json
{
  "downloaded_at": "ISO 8601 (UTC)",
  "target": "coco",
  "split": "val",
  "data_root": "data",
  "seed": 42,
  "files": [
    {
      "name": "val2017.zip",
      "url": "...",
      "size_bytes": 1234567890,
      "sha256": "...",
      "unzip_ok": true
    }
  ],
  "extracted": {
    "images_dir": "data/coco/val2017",
    "annotations_dir": "data/coco/annotations"
  },
  "val_images": 5000,
  "ann_files": 6,
  "instances_val2017_exists": true,
  "integrity_ok": true
}
```

`integrity_ok` 는 위 검증 모두 통과 시에만 true. 하나라도 실패면 false + `error_reasons: [...]` 추가.

### 6. 시드 의무 (CRITICAL)

`--seed` 는 본 step 에서 random 사용 없어도 인자로 받아 manifest 에 기록 (재현성 메타).

## Acceptance Criteria

```bash
# 다운로드 + 검증 실행 (한 번에)
python -m datasets.coco.download --target coco --split val --data-root data --seed 42

# 산출물 확인
ls data/coco/val2017 | wc -l                        # 5000
ls data/coco/annotations/instances_val2017.json     # 존재
ls runs/data-sanity-download-*/manifest.json | head -1  # 1 개

# 자동 검증 (execute.py 의 success_metric 과 동일)
jq -e '.val_images == 5000 and .ann_files >= 4 and .integrity_ok == true' \
   runs/data-sanity-download-*/manifest.json
```

## 검증 절차

1. 위 AC 명령을 순서대로 실행.
2. `runs/data-sanity-download-{ts}/manifest.json` 의 `integrity_ok == true` 확인.
3. `phases/data-sanity-coco/index.json` 의 step 0 status 갱신 — `completed` + `summary` ("val2017 5000장 + annotations 6 파일 / integrity OK"). 실패 시 `error` + `error_message`.

## 금지사항

- **detectron2 import 절대 금지.** 이유: CLAUDE.md CRITICAL — fm-det 의 존재 이유는 직접 재구현.
- **train2017.zip (~18GB) 을 본 step 에서 다운로드하지 마라.** 이유: 본 step 의 scope 는 val 만. train 은 P0 학습 phase 직전 별도 step.
- **`data/` 디렉터리를 `git add` 하지 마라.** 이유: gitignore 대상 + PreToolUse hook 차단. 가중치/데이터 강제 추가 사고 방지.
- **`scripts/` 폴더에 데이터 다운 스크립트를 두지 마라.** 이유: CLAUDE.md 규칙 — scripts/ 는 하네스 정본 전용. 진입점은 루트 평탄 (`datasets/coco/download.py`).
- **체크섬 미일치 시 검증 우회 (skip) 하지 마라.** 이유: 데이터 무결성은 모든 후속 학습/평가의 전제.
