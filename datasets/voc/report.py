"""PASCAL VOC data-sanity-voc phase 의 산출물 묶기 → report.md."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.voc.sanity import SPLIT_MAP, VOC_CLASSES  # noqa: E402


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _latest(prefix: str) -> Path:
    cands = sorted((ROOT / "runs").glob(f"{prefix}-*"))
    assert cands, f"no runs/{prefix}-* found"
    return cands[-1]


def _percentile_row(name: str, d: dict) -> str:
    def _fmt(v):
        if v is None:
            return "-"
        return f"{v:.2f}" if isinstance(v, float) else str(v)
    cells = " | ".join(_fmt(d.get(k)) for k in ("mean", "median", "p10", "p90", "min", "max"))
    return f"| {name} | {cells} |"


def compose(download_run: Path, analyze_run: Path, vis_run: Path,
            run_dir: Path, seed: int) -> Path:
    manifest = json.loads((download_run / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((analyze_run / "stats.json").read_text(encoding="utf-8"))
    vis_manifest = json.loads((vis_run / "vis_manifest.json").read_text(encoding="utf-8"))

    download_rows = ["| 파일 | 크기 (MB) | SHA256 (12자) | untar |",
                     "|------|----------|--------------|-------|"]
    for f in manifest["files"]:
        size_mb = f["size_bytes"] / (1024 * 1024)
        download_rows.append(
            f"| {f['name']} | {size_mb:.1f} | `{f['sha256'][:12]}` | {'✅' if f['untar_ok'] else '❌'} |"
        )

    cnt = manifest["image_counts"]
    exp = manifest["image_counts_expected"]
    counts_rows = ["| split | 측정 | DATA_CARD 기댓값 | 일치 |",
                   "|-------|------|----------------|------|"]
    for k in ("voc07-trainval", "voc07-test", "voc12-trainval", "voc-trainval"):
        c = cnt.get(k, 0)
        e = exp.get(k, "-")
        ok = "✅" if c == e else "❌"
        counts_rows.append(f"| {k} | {c} | {e} | {ok} |")

    # 분포 표 (split 별 num_ann + bbox stats)
    splits_rows = ["| split | images | annotations | difficult | bbox xyxy valid | class id valid |",
                   "|-------|--------|-------------|-----------|-----------------|----------------|"]
    for k in ("voc07-trainval", "voc07-test", "voc12-trainval"):
        s = stats["splits"][k]
        splits_rows.append(
            f"| {k} | {s['num_images']} | {s['num_annotations']} | {s['num_difficult']} "
            f"| {s['bbox_xyxy_valid']} | {s['class_id_valid']} |"
        )

    # 비교 시각화는 vis_run 의 한 split 만 (default voc07-trainval)
    vis_split = vis_manifest.get("split", "voc07-trainval")
    vis_rel = vis_run.name

    # 클래스 top/bottom 5 — voc07-trainval 기준
    cd = stats["splits"]["voc07-trainval"]["class_distribution"]
    top = sorted(cd.items(), key=lambda x: -x[1])[:5]
    bot = sorted(cd.items(), key=lambda x: -x[1])[-5:][::-1]
    cls_rows = ["| 순위 | 클래스 | 개수 (voc07-trainval) |", "|------|--------|-------|"]
    for i, (n, c) in enumerate(top, 1): cls_rows.append(f"| top {i} | {n} | {c} |")
    for i, (n, c) in enumerate(bot, 1): cls_rows.append(f"| bottom {i} | {n} | {c} |")

    bbox_table = "\n".join([
        "| 통계 | mean | median | p10 | p90 | min | max |",
        "|------|------|--------|-----|-----|-----|-----|",
        _percentile_row(f"bbox area (px²) — {vis_split}", stats["splits"][vis_split]["bbox_area_stats"]),
        _percentile_row(f"aspect ratio (w/h) — {vis_split}", stats["splits"][vis_split]["bbox_aspect_ratio_stats"]),
    ])
    size_table = "\n".join([
        "| 통계 | mean | median | p10 | p90 | min | max |",
        "|------|------|--------|-----|-----|-----|-----|",
        _percentile_row(f"image width (px) — {vis_split}", stats["splits"][vis_split]["image_size_stats"]["width"]),
        _percentile_row(f"image height (px) — {vis_split}", stats["splits"][vis_split]["image_size_stats"]["height"]),
    ])

    body = f"""# Data Sanity Report — PASCAL VOC 2007 + 2012

- **생성일**: {datetime.now(timezone.utc).isoformat()}
- **데이터셋**: PASCAL VOC 2007 + 2012 (`{len(VOC_CLASSES)}` classes)
- **시드**: {seed}
- **참조 phase**: `phases/data-sanity-voc/`
- **참조 runs**: download=`{download_run.name}` / analyze=`{analyze_run.name}` / vis=`{vis_run.name}` (split=`{vis_split}`)

## 다운로드 결과

{chr(10).join(download_rows)}

- integrity_ok: **{manifest['integrity_ok']}**

## DATA_CARD 분포 일치

{chr(10).join(counts_rows)}

## split 별 분포 통계

{chr(10).join(splits_rows)}

- voc-trainval 합본 (07+12): **{stats['voc_trainval_combined']['num_images']}** 이미지 / **{stats['voc_trainval_combined']['num_annotations']}** annotations

### 박스 통계 (`{vis_split}`)

{bbox_table}

### 이미지 해상도 (`{vis_split}`)

{size_table}

### 클래스 분포 top/bottom 5 (`voc07-trainval`)

{chr(10).join(cls_rows)}

## 시각화 (`{vis_split}`)

![class_dist](../{vis_rel}/figs/class_dist.png)
![bbox_size](../{vis_rel}/figs/bbox_size.png)
![image_size](../{vis_rel}/figs/image_size.png)
![samples](../{vis_rel}/figs/samples.png)

## 다음 단계

- [ ] VOC mAP@0.5 평가 코드 (`evals/voc.py`) — P0 VOC baseline 학습 직전
- [ ] COCO val 과의 데이터 특성 비교 노트 (선택)
- [ ] 본 phase 결과로 `datasets/voc/dataset.py` (PyTorch Dataset) 인터페이스 확정 — P0 VOC baseline 시작 자료
"""

    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "report.md"
    out.write_text(body, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Compose data-sanity-voc report.")
    ap.add_argument("--download-run", type=Path, default=None)
    ap.add_argument("--analyze-run", type=Path, default=None)
    ap.add_argument("--vis-run", type=Path, default=None)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    download_run = args.download_run or _latest("data-sanity-voc-download")
    analyze_run = args.analyze_run or _latest("data-sanity-voc-analyze")
    vis_run = args.vis_run or _latest("data-sanity-voc-vis")

    run_dir = ROOT / "runs" / f"data-sanity-voc-report-{_now_tag()}"
    out = compose(download_run, analyze_run, vis_run, run_dir, args.seed)
    print(f"\nreport → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
