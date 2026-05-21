"""PASCAL VOC 2007/2012 분포·박스 통계 분석.

xml.etree (stdlib) 로 Annotations/*.xml 파싱. detectron2 금지.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

SPLIT_MAP = {
    "voc07-trainval": ("2007", "trainval"),
    "voc07-test": ("2007", "test"),
    "voc12-trainval": ("2012", "trainval"),
}


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


def _parse_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    W = int(size.find("width").text)
    H = int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        diff_el = obj.find("difficult")
        difficult = int(diff_el.text) if diff_el is not None else 0
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        boxes.append({
            "name": name, "difficult": difficult,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
        })
    return {"width": W, "height": H, "boxes": boxes}


def _analyze_split(devkit: Path, year: str, split: str) -> dict:
    voc_year = devkit / f"VOC{year}"
    ann_dir = voc_year / "Annotations"
    ids_file = voc_year / "ImageSets" / "Main" / f"{split}.txt"
    img_ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    class_dist = {c: 0 for c in VOC_CLASSES}
    areas, ars, widths, heights = [], [], [], []
    bbox_xyxy_valid = True
    class_id_valid = True
    invalid_examples: list[dict] = []
    n_ann = 0
    n_difficult = 0
    n_images_with_ann = 0

    valid_cls = set(VOC_CLASSES)
    for img_id in img_ids:
        xml = ann_dir / f"{img_id}.xml"
        info = _parse_xml(xml)
        W, H = info["width"], info["height"]
        widths.append(W)
        heights.append(H)
        if info["boxes"]:
            n_images_with_ann += 1
        for bx in info["boxes"]:
            n_ann += 1
            if bx["difficult"]:
                n_difficult += 1
            if bx["name"] not in valid_cls:
                class_id_valid = False
                if len(invalid_examples) < 5:
                    invalid_examples.append({"image_id": img_id, "reason": f"invalid class {bx['name']}"})
                continue
            class_dist[bx["name"]] += 1
            x1, y1, x2, y2 = bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"]
            if not (x1 >= 0 and y1 >= 0 and x2 > x1 and y2 > y1 and x2 <= W + 1e-3 and y2 <= H + 1e-3):
                bbox_xyxy_valid = False
                if len(invalid_examples) < 5:
                    invalid_examples.append({
                        "image_id": img_id, "bbox_xyxy": [x1, y1, x2, y2],
                        "image_wh": [W, H], "reason": "bbox out of bounds",
                    })
            w, h = x2 - x1, y2 - y1
            if w > 0 and h > 0:
                areas.append(w * h)
                ars.append(w / h)

    return {
        "num_images": len(img_ids),
        "num_annotations": n_ann,
        "num_difficult": n_difficult,
        "num_images_with_ann": n_images_with_ann,
        "num_classes": len(VOC_CLASSES),
        "bbox_xyxy_valid": bbox_xyxy_valid,
        "class_id_valid": class_id_valid,
        "class_distribution": {n: class_dist[n] for n in VOC_CLASSES},
        "bbox_area_stats": _percentiles(np.asarray(areas)),
        "bbox_aspect_ratio_stats": _percentiles(np.asarray(ars)),
        "image_size_stats": {
            "width": _percentiles(np.asarray(widths, dtype=np.float64)),
            "height": _percentiles(np.asarray(heights, dtype=np.float64)),
        },
        "invalid_examples": invalid_examples,
    }


def analyze(data_root: Path, run_dir: Path, seed: int) -> dict:
    devkit = data_root / "voc" / "VOCdevkit"
    assert devkit.exists(), f"devkit not found: {devkit}"

    per_split = {}
    for name, (year, sp) in SPLIT_MAP.items():
        per_split[name] = _analyze_split(devkit, year, sp)

    # 합본: voc-trainval = voc07-trainval + voc12-trainval (sum)
    voc_tv = per_split["voc07-trainval"]["num_images"] + per_split["voc12-trainval"]["num_images"]
    voc_tv_ann = per_split["voc07-trainval"]["num_annotations"] + per_split["voc12-trainval"]["num_annotations"]

    stats = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.relative_to(ROOT)) if data_root.is_relative_to(ROOT) else str(data_root),
        "seed": seed,
        "splits": per_split,
        "voc_trainval_combined": {
            "num_images": voc_tv,
            "num_annotations": voc_tv_ann,
        },
        # 전체 검증 요약 (success_metric 용)
        "bbox_xyxy_valid": all(s["bbox_xyxy_valid"] for s in per_split.values()),
        "class_id_valid": all(s["class_id_valid"] for s in per_split.values()),
        "num_classes": len(VOC_CLASSES),
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze PASCAL VOC distribution.")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / f"data-sanity-voc-analyze-{_now_tag()}"
    stats = analyze(args.data_root, run_dir, args.seed)
    print(f"\nstats → {run_dir / 'stats.json'}")
    summary = {
        "num_classes": stats["num_classes"],
        "bbox_xyxy_valid": stats["bbox_xyxy_valid"],
        "class_id_valid": stats["class_id_valid"],
        "voc07-trainval": stats["splits"]["voc07-trainval"]["num_images"],
        "voc07-test": stats["splits"]["voc07-test"]["num_images"],
        "voc12-trainval": stats["splits"]["voc12-trainval"]["num_images"],
        "voc-trainval-combined": stats["voc_trainval_combined"]["num_images"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if (stats["bbox_xyxy_valid"] and stats["class_id_valid"]) else 1


if __name__ == "__main__":
    sys.exit(main())
