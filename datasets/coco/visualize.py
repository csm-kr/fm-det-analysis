"""COCO val 분포·박스·이미지 크기 시각화 + GT bbox overlay 샘플.

matplotlib (Agg backend) + Pillow. detectron2 visualizer 금지.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO

ROOT = Path(__file__).resolve().parents[2]


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _latest_dir(prefix: str) -> Path:
    cands = sorted((ROOT / "runs").glob(f"{prefix}-*"))
    assert cands, f"no runs/{prefix}-* found"
    return cands[-1]


def _hsv_colors(n: int) -> list[tuple[int, int, int]]:
    """HSV cycle → RGB uint8."""
    import colorsys
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 0.8, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _fig_class_dist(stats: dict, out_path: Path) -> None:
    items = sorted(
        ((v["name"], v["count"]) for v in stats["class_distribution"].values()),
        key=lambda x: -x[1],
    )
    names, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)
    ax.bar(range(len(counts)), counts, color="#1f77b4")
    ax.set_yscale("log")
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(names, rotation=75, fontsize=7)
    ax.set_ylabel("annotations (log)")
    ax.set_title(f"COCO val — class distribution (80 classes, sorted desc)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fig_bbox_size(stats: dict, areas: np.ndarray, ars: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    a_stats = stats["bbox_area_stats"]
    bins = np.logspace(np.log10(max(a_stats["min"], 1.0)), np.log10(a_stats["max"]), 100)
    axes[0].hist(areas, bins=bins, color="#2ca02c", alpha=0.8)
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    for k, c in (("p10", "tab:gray"), ("median", "tab:red"), ("p90", "tab:gray")):
        axes[0].axvline(a_stats[k], linestyle="--", color=c, alpha=0.7, label=f"{k}={a_stats[k]:.0f}")
    axes[0].set_xlabel("bbox area (px²)"); axes[0].set_ylabel("count")
    axes[0].set_title("bbox area distribution (log-log)"); axes[0].legend()
    axes[0].grid(linestyle=":", alpha=0.4)

    ar_clip = np.clip(ars, 0, 5)
    ar_stats = stats["bbox_aspect_ratio_stats"]
    axes[1].hist(ar_clip, bins=50, range=(0, 5), color="#ff7f0e", alpha=0.8)
    for k, c in (("p10", "tab:gray"), ("median", "tab:red"), ("p90", "tab:gray")):
        v = ar_stats[k]
        if v is not None and 0 <= v <= 5:
            axes[1].axvline(v, linestyle="--", color=c, alpha=0.7, label=f"{k}={v:.2f}")
    axes[1].set_xlabel("aspect ratio (w/h, clipped at 5)"); axes[1].set_ylabel("count")
    axes[1].set_title("bbox aspect ratio distribution"); axes[1].legend()
    axes[1].grid(linestyle=":", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fig_image_size(widths: np.ndarray, heights: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    h2d, xe, ye, im = ax.hist2d(
        widths, heights, bins=50,
        norm=matplotlib.colors.LogNorm(),
        cmap="viridis",
    )
    ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)")
    ax.set_title("COCO val — image size 2D histogram (log color)")
    # aspect 가이드선
    for ratio, label in ((1.0, "1:1"), (4 / 3, "4:3"), (16 / 9, "16:9")):
        wmax = float(widths.max())
        xs = np.linspace(0, wmax, 100)
        ax.plot(xs, xs / ratio, "--", color="white", alpha=0.5, linewidth=0.8, label=label)
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="count (log)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _fig_samples(coco: COCO, images_dir: Path, seed: int, out_path: Path) -> list[int]:
    rng = np.random.default_rng(seed)
    img_ids = sorted(coco.getImgIds())
    # ann 없는 이미지 빼고 sampling — 시각화 의미를 위해
    img_ids_with_ann = [i for i in img_ids if coco.getAnnIds(imgIds=[i])]
    chosen = rng.choice(img_ids_with_ann, size=9, replace=False).tolist()

    cat_ids = sorted(coco.getCatIds())
    cat_names = {c["id"]: c["name"] for c in coco.loadCats(cat_ids)}
    palette = {cid: c for cid, c in zip(cat_ids, _hsv_colors(len(cat_ids)))}

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    fig, axes = plt.subplots(3, 3, figsize=(15, 15), dpi=120)
    for ax, img_id in zip(axes.flat, chosen):
        img_info = coco.loadImgs([int(img_id)])[0]
        img_path = images_dir / img_info["file_name"]
        pil = Image.open(img_path).convert("RGB")
        # 큰 이미지는 리사이즈 (긴 변 800)
        scale = min(800 / pil.width, 800 / pil.height, 1.0)
        if scale < 1.0:
            new_w, new_h = int(pil.width * scale), int(pil.height * scale)
            pil = pil.resize((new_w, new_h), Image.BILINEAR)
        else:
            scale = 1.0
        draw = ImageDraw.Draw(pil)

        ann_ids = coco.getAnnIds(imgIds=[int(img_id)])
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            x, y, w, h = a["bbox"]
            x, y, w, h = x * scale, y * scale, w * scale, h * scale
            color = palette[a["category_id"]]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            label = cat_names[a["category_id"]]
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            draw.rectangle([x, y - th - 2, x + tw + 4, y], fill=color)
            draw.text((x + 2, y - th - 2), label, fill="white", font=font)

        ax.imshow(np.asarray(pil))
        ax.set_title(f"image_id={img_id} | {len(anns)} boxes", fontsize=10)
        ax.axis("off")

    fig.suptitle(f"COCO val — random 9 (seed={seed})", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return [int(x) for x in chosen]


def visualize(split: str, data_root: Path, stats_json: Path, run_dir: Path, seed: int) -> dict:
    assert split in ("val", "train")
    stats = json.loads(stats_json.read_text(encoding="utf-8"))

    ann_path = data_root / "coco" / "annotations" / f"instances_{split}2017.json"
    images_dir = data_root / "coco" / f"{split}2017"
    coco = COCO(str(ann_path))

    # bbox 통계용 raw 값 재계산 (stats.json 에 raw 없음 — 분포만 있음)
    img_ids = coco.getImgIds()
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_ids))
    areas, ars = [], []
    for a in anns:
        x, y, w, h = a["bbox"]
        if w > 0 and h > 0:
            areas.append(w * h)
            ars.append(w / h)
    areas = np.asarray(areas)
    ars = np.asarray(ars)

    widths = np.asarray([img["width"] for img in coco.loadImgs(img_ids)])
    heights = np.asarray([img["height"] for img in coco.loadImgs(img_ids)])

    figs_dir = run_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    _fig_class_dist(stats, figs_dir / "class_dist.png")
    _fig_bbox_size(stats, areas, ars, figs_dir / "bbox_size.png")
    _fig_image_size(widths, heights, figs_dir / "image_size.png")
    sample_ids = _fig_samples(coco, images_dir, seed, figs_dir / "samples.png")

    vis_manifest = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "stats_json": str(stats_json.relative_to(ROOT)) if stats_json.is_relative_to(ROOT) else str(stats_json),
        "seed": seed,
        "sample_image_ids": sample_ids,
        "figures": [str(p.relative_to(run_dir)) for p in sorted(figs_dir.glob("*.png"))],
    }
    (run_dir / "vis_manifest.json").write_text(
        json.dumps(vis_manifest, indent=2), encoding="utf-8"
    )
    return vis_manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize COCO val distribution.")
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--stats-json", type=Path, default=None,
                    help="없으면 runs/data-sanity-analyze-* 최근 자동 선택")
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    stats_json = args.stats_json or (_latest_dir("data-sanity-analyze") / "stats.json")
    run_dir = ROOT / "runs" / f"data-sanity-vis-{_now_tag()}"
    vis = visualize(args.split, args.data_root, stats_json, run_dir, args.seed)
    print(f"\nfigs → {run_dir / 'figs'}")
    print(json.dumps(vis, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
