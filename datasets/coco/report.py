"""data-sanity-coco phase 의 3 step 산출물을 묶어 report.md 생성."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _latest(prefix: str) -> Path:
    cands = sorted((ROOT / "runs").glob(f"{prefix}-*"))
    assert cands, f"no runs/{prefix}-* found"
    return cands[-1]


def _kv_table(d: dict, keys: list[str]) -> str:
    rows = ["| 항목 | 값 |", "|------|-----|"]
    for k in keys:
        rows.append(f"| {k} | {d.get(k)} |")
    return "\n".join(rows)


def _stats_table(d: dict) -> str:
    rows = ["| 통계 | mean | median | p10 | p90 | min | max |",
            "|------|------|--------|-----|-----|-----|-----|"]

    def _fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    for k in ("mean", "median", "p10", "p90", "min", "max"):
        pass
    return rows


def _percentile_row(name: str, d: dict) -> str:
    def _fmt(v):
        if v is None:
            return "-"
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    cells = " | ".join(_fmt(d.get(k)) for k in ("mean", "median", "p10", "p90", "min", "max"))
    return f"| {name} | {cells} |"


def _top_bottom_classes(stats: dict, n: int = 5) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    items = [(v["name"], v["count"]) for v in stats["class_distribution"].values()]
    items.sort(key=lambda x: -x[1])
    return items[:n], items[-n:][::-1]


def compose(download_run: Path, analyze_run: Path, vis_run: Path,
            run_dir: Path, seed: int) -> Path:
    manifest = json.loads((download_run / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((analyze_run / "stats.json").read_text(encoding="utf-8"))
    vis_manifest = json.loads((vis_run / "vis_manifest.json").read_text(encoding="utf-8"))

    download_table_rows = ["| 파일 | 크기 (MB) | SHA256 (12자) | unzip |",
                            "|------|----------|--------------|-------|"]
    for f in manifest["files"]:
        size_mb = f["size_bytes"] / (1024 * 1024)
        download_table_rows.append(
            f"| {f['name']} | {size_mb:.1f} | `{f['sha256'][:12]}` | {'✅' if f['unzip_ok'] else '❌'} |"
        )

    dist_overview = _kv_table(stats, [
        "num_images", "num_annotations", "num_classes",
        "num_images_with_ann", "num_images_no_ann",
        "bbox_xyxy_valid", "class_id_valid",
    ])

    bbox_table = "\n".join([
        "| 통계 | mean | median | p10 | p90 | min | max |",
        "|------|------|--------|-----|-----|-----|-----|",
        _percentile_row("bbox area (px²)", stats["bbox_area_stats"]),
        _percentile_row("aspect ratio (w/h)", stats["bbox_aspect_ratio_stats"]),
    ])

    size_table = "\n".join([
        "| 통계 | mean | median | p10 | p90 | min | max |",
        "|------|------|--------|-----|-----|-----|-----|",
        _percentile_row("image width (px)", stats["image_size_stats"]["width"]),
        _percentile_row("image height (px)", stats["image_size_stats"]["height"]),
    ])

    top, bottom = _top_bottom_classes(stats, 5)
    cls_rows = ["| 순위 | 클래스 | 개수 |", "|------|--------|------|"]
    for i, (n, c) in enumerate(top, 1):
        cls_rows.append(f"| top {i} | {n} | {c} |")
    for i, (n, c) in enumerate(bottom, 1):
        cls_rows.append(f"| bottom {i} | {n} | {c} |")

    # DATA_CARD 일치 여부
    match_rows = ["| 항목 | DATA_CARD | 측정 | 일치 |",
                  "|------|-----------|------|------|",
                  f"| val 이미지 수 | 5,000 | {stats['num_images']} | {'✅' if stats['num_images'] == 5000 else '❌'} |",
                  f"| 클래스 수 | 80 | {stats['num_classes']} | {'✅' if stats['num_classes'] == 80 else '❌'} |",
                  f"| integrity OK | true | {manifest['integrity_ok']} | {'✅' if manifest['integrity_ok'] else '❌'} |"]

    # 상대 경로 figure
    vis_rel = vis_run.name  # runs/{vis_rel}/figs/*.png

    body = f"""# Data Sanity Report — COCO 2017 val

- **생성일**: {datetime.now(timezone.utc).isoformat()}
- **데이터셋**: COCO 2017
- **분할**: val
- **시드**: {seed}
- **참조 phase**: `phases/data-sanity-coco/`
- **참조 runs**: download=`{download_run.name}` / analyze=`{analyze_run.name}` / vis=`{vis_run.name}`

## 다운로드 결과

{chr(10).join(download_table_rows)}

- val 이미지 수: **{manifest['val_images']}**
- annotations 파일 수: **{manifest['ann_files']}**
- instances_val2017.json 존재: **{manifest['instances_val2017_exists']}**
- integrity_ok: **{manifest['integrity_ok']}**

## 분포 통계

{dist_overview}

### 박스 통계

{bbox_table}

### 이미지 해상도

{size_table}

### 클래스 분포 상위/하위 5

{chr(10).join(cls_rows)}

## 시각화

![class_dist](../{vis_rel}/figs/class_dist.png)
![bbox_size](../{vis_rel}/figs/bbox_size.png)
![image_size](../{vis_rel}/figs/image_size.png)
![samples](../{vis_rel}/figs/samples.png)

## DATA_CARD 표와의 일치 여부

{chr(10).join(match_rows)}

## 다음 단계

- [ ] train2017 (18GB) 다운로드 phase 추가 — P0 baseline 학습 직전
- [ ] PASCAL VOC 다운로드는 별도 phase (`data-sanity-voc`) — P3 시점
- [ ] 본 phase 의 결과로 `datasets/coco.py` GT 로딩 인터페이스 확정 (P0 baseline 시작 자료)
"""

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "report.md"
    out.write_text(body, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose data-sanity-coco report.")
    ap.add_argument("--download-run", type=Path, default=None)
    ap.add_argument("--analyze-run", type=Path, default=None)
    ap.add_argument("--vis-run", type=Path, default=None)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    download_run = args.download_run or _latest("data-sanity-download")
    analyze_run = args.analyze_run or _latest("data-sanity-analyze")
    vis_run = args.vis_run or _latest("data-sanity-vis")

    run_dir = ROOT / "runs" / f"data-sanity-report-{_now_tag()}"
    out = compose(download_run, analyze_run, vis_run, run_dir, args.seed)
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
