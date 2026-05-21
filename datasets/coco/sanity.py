"""COCO val 분포·결측·박스 통계 분석.

pycocotools 만 사용 (detectron2 금지 — CLAUDE.md CRITICAL).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _percentiles(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def analyze(split: str, data_root: Path, run_dir: Path, seed: int) -> dict:
    assert split in ("val", "train"), f"unsupported split: {split}"
    ann_path = data_root / "coco" / "annotations" / f"instances_{split}2017.json"
    assert ann_path.exists(), f"annotation not found: {ann_path}"

    coco = COCO(str(ann_path))
    cat_ids = sorted(coco.getCatIds())
    cats = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    img_ids = coco.getImgIds()

    # 클래스 분포
    class_distribution: dict[int, int] = {cid: 0 for cid in cat_ids}
    # bbox 통계용 어레이
    areas: list[float] = []
    aspect_ratios: list[float] = []
    # box 유효성
    bbox_xyxy_valid = True
    class_id_valid = True
    invalid_examples: list[dict] = []

    # 이미지 크기 통계
    widths: list[int] = []
    heights: list[int] = []
    img_info_by_id = {img["id"]: img for img in coco.loadImgs(img_ids)}
    for img in img_info_by_id.values():
        widths.append(int(img["width"]))
        heights.append(int(img["height"]))

    # 어노테이션 순회
    ann_ids = coco.getAnnIds(imgIds=img_ids)
    anns = coco.loadAnns(ann_ids)
    num_images_with_ann = len({a["image_id"] for a in anns})
    num_images_no_ann = len(img_ids) - num_images_with_ann

    valid_cat_set = set(cat_ids)
    for a in anns:
        cid = a["category_id"]
        if cid not in valid_cat_set:
            class_id_valid = False
            if len(invalid_examples) < 5:
                invalid_examples.append({"ann_id": a["id"], "reason": f"invalid cat_id {cid}"})
            continue
        class_distribution[cid] += 1

        x, y, w, h = a["bbox"]
        x2, y2 = x + w, y + h
        img = img_info_by_id[a["image_id"]]
        W, H = img["width"], img["height"]
        if not (x >= 0 and y >= 0 and x2 > x and y2 > y and x2 <= W + 1e-3 and y2 <= H + 1e-3):
            bbox_xyxy_valid = False
            if len(invalid_examples) < 5:
                invalid_examples.append({
                    "ann_id": a["id"], "image_id": a["image_id"],
                    "bbox_xywh": [x, y, w, h], "image_wh": [W, H],
                    "reason": "bbox out of image bounds or non-positive",
                })
        if w > 0 and h > 0:
            areas.append(float(w * h))
            aspect_ratios.append(float(w / h))

    areas_a = np.asarray(areas, dtype=np.float64)
    ar_a = np.asarray(aspect_ratios, dtype=np.float64)
    widths_a = np.asarray(widths, dtype=np.float64)
    heights_a = np.asarray(heights, dtype=np.float64)

    # class_distribution: int 키 → str 키 (jq 호환 + json 안정)
    class_distribution_named = {
        str(cid): {"name": cats[cid], "count": class_distribution[cid]}
        for cid in cat_ids
    }

    stats = {
        "split": split,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.relative_to(ROOT)) if data_root.is_relative_to(ROOT) else str(data_root),
        "seed": seed,
        "num_images": len(img_ids),
        "num_annotations": len(anns),
        "num_classes": len(cat_ids),
        "num_images_with_ann": num_images_with_ann,
        "num_images_no_ann": num_images_no_ann,
        "class_distribution": class_distribution_named,
        "bbox_xyxy_valid": bbox_xyxy_valid,
        "class_id_valid": class_id_valid,
        "invalid_examples": invalid_examples,
        "bbox_area_stats": _percentiles(areas_a),
        "bbox_aspect_ratio_stats": _percentiles(ar_a),
        "image_size_stats": {
            "width": _percentiles(widths_a),
            "height": _percentiles(heights_a),
        },
    }

    # 자체 일관성 — class_distribution 합 == num_annotations (잘못된 cat_id 어노 제외)
    dist_sum = sum(v["count"] for v in class_distribution_named.values())
    stats["class_distribution_sum_matches"] = (dist_sum == len(anns)) if class_id_valid else None

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze COCO val distribution.")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / f"data-sanity-analyze-{_now_tag()}"
    stats = analyze(args.split, args.data_root, run_dir, args.seed)
    print(f"\nstats → {run_dir / 'stats.json'}")
    keys = ("num_images", "num_annotations", "num_classes",
            "num_images_with_ann", "num_images_no_ann",
            "bbox_xyxy_valid", "class_id_valid")
    print(json.dumps({k: stats[k] for k in keys}, indent=2))
    return 0 if (stats["bbox_xyxy_valid"] and stats["class_id_valid"]) else 1


if __name__ == "__main__":
    sys.exit(main())
