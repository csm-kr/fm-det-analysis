"""PASCAL VOC 시각화 — 4 figure (split 별)."""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from datasets.voc.sanity import SPLIT_MAP, VOC_CLASSES, _parse_xml  # noqa: E402


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _latest_dir(prefix: str) -> Path:
    cands = sorted((ROOT / "runs").glob(f"{prefix}-*"))
    assert cands, f"no runs/{prefix}-* found"
    return cands[-1]


def _hsv_colors(n: int) -> list[tuple[int, int, int]]:
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.8, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _load_split(devkit: Path, year: str, split: str):
    voc_year = devkit / f"VOC{year}"
    ids = [l.strip() for l in (voc_year / "ImageSets" / "Main" / f"{split}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    return voc_year, ids


def _fig_class_dist(split_stats: dict, out_path: Path, split: str) -> None:
    items = sorted(split_stats["class_distribution"].items(), key=lambda x: -x[1])
    names, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
    ax.bar(range(len(counts)), counts, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(names, rotation=60, fontsize=8)
    ax.set_ylabel("annotations (log)")
    ax.set_title(f"VOC {split} — class distribution (20 classes, sorted desc)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _fig_bbox_size(split_stats: dict, areas: np.ndarray, ars: np.ndarray, out_path: Path, split: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    a_stats = split_stats["bbox_area_stats"]
    bins = np.logspace(np.log10(max(a_stats["min"], 1.0)), np.log10(a_stats["max"]), 100)
    axes[0].hist(areas, bins=bins, color="#2ca02c", alpha=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    for k, c in (("p10", "tab:gray"), ("median", "tab:red"), ("p90", "tab:gray")):
        axes[0].axvline(a_stats[k], linestyle="--", color=c, alpha=0.7, label=f"{k}={a_stats[k]:.0f}")
    axes[0].set_xlabel("bbox area (px²)"); axes[0].set_ylabel("count")
    axes[0].set_title(f"VOC {split} — bbox area"); axes[0].legend(); axes[0].grid(linestyle=":", alpha=0.4)

    ar_stats = split_stats["bbox_aspect_ratio_stats"]
    axes[1].hist(np.clip(ars, 0, 5), bins=50, range=(0, 5), color="#ff7f0e", alpha=0.8)
    for k, c in (("p10", "tab:gray"), ("median", "tab:red"), ("p90", "tab:gray")):
        v = ar_stats[k]
        if v is not None and 0 <= v <= 5:
            axes[1].axvline(v, linestyle="--", color=c, alpha=0.7, label=f"{k}={v:.2f}")
    axes[1].set_xlabel("aspect ratio (w/h, clipped at 5)"); axes[1].set_ylabel("count")
    axes[1].set_title(f"VOC {split} — aspect ratio"); axes[1].legend(); axes[1].grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _fig_image_size(widths: np.ndarray, heights: np.ndarray, out_path: Path, split: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    _, _, _, im = ax.hist2d(widths, heights, bins=50, norm=matplotlib.colors.LogNorm(), cmap="viridis")
    ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
    ax.set_title(f"VOC {split} — image size 2D hist (log color)")
    wmax = float(widths.max())
    xs = np.linspace(0, wmax, 100)
    for ratio, label in ((1.0, "1:1"), (4 / 3, "4:3"), (16 / 9, "16:9")):
        ax.plot(xs, xs / ratio, "--", color="white", alpha=0.5, linewidth=0.8, label=label)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="count (log)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


def _fig_samples(devkit: Path, year: str, split: str, ids: list[str], seed: int, out_path: Path) -> list[str]:
    rng = np.random.default_rng(seed)
    voc_year = devkit / f"VOC{year}"
    ann_dir = voc_year / "Annotations"
    img_dir = voc_year / "JPEGImages"
    ids_with_ann = [i for i in ids if _parse_xml(ann_dir / f"{i}.xml")["boxes"]]
    chosen = rng.choice(ids_with_ann, size=9, replace=False).tolist()

    palette = {c: col for c, col in zip(VOC_CLASSES, _hsv_colors(len(VOC_CLASSES)))}
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=120)
    for ax, img_id in zip(axes.flat, chosen):
        pil = Image.open(img_dir / f"{img_id}.jpg").convert("RGB")
        scale = min(800 / pil.width, 800 / pil.height, 1.0)
        if scale < 1.0:
            pil = pil.resize((int(pil.width * scale), int(pil.height * scale)), Image.BILINEAR)
        else:
            scale = 1.0
        draw = ImageDraw.Draw(pil)
        info = _parse_xml(ann_dir / f"{img_id}.xml")
        for bx in info["boxes"]:
            if bx["name"] not in palette:
                continue
            color = palette[bx["name"]]
            x1, y1 = bx["xmin"] * scale, bx["ymin"] * scale
            x2, y2 = bx["xmax"] * scale, bx["ymax"] * scale
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            tw, th = draw.textbbox((0, 0), bx["name"], font=font)[2:]
            draw.rectangle([x1, y1 - th - 2, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - th - 2), bx["name"], fill="white", font=font)
        ax.imshow(np.asarray(pil))
        ax.set_title(f"id={img_id} | {len(info['boxes'])} boxes", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"VOC {split} — random 9 (seed={seed})", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)
    return chosen


def visualize(split: str, data_root: Path, stats_json: Path, run_dir: Path, seed: int) -> dict:
    assert split in SPLIT_MAP, f"unknown split: {split}"
    year, sp = SPLIT_MAP[split]
    stats = json.loads(stats_json.read_text(encoding="utf-8"))
    split_stats = stats["splits"][split]

    devkit = data_root / "voc" / "VOCdevkit"
    voc_year, ids = _load_split(devkit, year, sp)
    ann_dir = voc_year / "Annotations"

    # raw 값 재계산 (시각화용)
    areas, ars, widths, heights = [], [], [], []
    for i in ids:
        info = _parse_xml(ann_dir / f"{i}.xml")
        widths.append(info["width"]); heights.append(info["height"])
        for bx in info["boxes"]:
            w = bx["xmax"] - bx["xmin"]
            h = bx["ymax"] - bx["ymin"]
            if w > 0 and h > 0:
                areas.append(w * h); ars.append(w / h)
    areas = np.asarray(areas); ars = np.asarray(ars)
    widths = np.asarray(widths, dtype=np.float64); heights = np.asarray(heights, dtype=np.float64)

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _fig_class_dist(split_stats, figs_dir / "class_dist.png", split)
    _fig_bbox_size(split_stats, areas, ars, figs_dir / "bbox_size.png", split)
    _fig_image_size(widths, heights, figs_dir / "image_size.png", split)
    sample_ids = _fig_samples(devkit, year, split, ids, seed, figs_dir / "samples.png")

    vis_manifest = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "stats_json": str(stats_json.relative_to(ROOT)) if stats_json.is_relative_to(ROOT) else str(stats_json),
        "seed": seed,
        "sample_image_ids": sample_ids,
        "figures": [str(p.relative_to(run_dir)) for p in sorted(figs_dir.glob("*.png"))],
    }
    (run_dir / "vis_manifest.json").write_text(json.dumps(vis_manifest, indent=2), encoding="utf-8")
    return vis_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize PASCAL VOC split.")
    ap.add_argument("--split", default="voc07-trainval", choices=list(SPLIT_MAP.keys()))
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--stats-json", type=Path, default=None)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    stats_json = args.stats_json or (_latest_dir("data-sanity-voc-analyze") / "stats.json")
    run_dir = ROOT / "runs" / f"data-sanity-voc-vis-{_now_tag()}"
    vis = visualize(args.split, args.data_root, stats_json, run_dir, args.seed)
    print(f"\nfigs → {run_dir / 'figs'}")
    print(json.dumps(vis, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
