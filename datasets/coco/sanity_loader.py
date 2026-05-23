"""COCO Dataset + DataLoader 1-batch sanity.

`build_coco_loader` 가 (B,3,H,W) tensor + list[target] 을 정상 산출하는지 검증.
cat_idx 가 [0, 80) 범위 안인지, num_targets 가 양수인지 확인.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from datasets.coco.dataset import build_coco_loader

ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 80


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(split: str, batch_size: int, seed: int, run_dir: Path) -> dict:
    assert split == "val", f"sanity_loader: only val split supported, got {split}"
    _seed_all(seed)

    cfg = OmegaConf.load(ROOT / "configs" / "data" / "coco.yaml")
    cfg.batch_size = batch_size
    cfg.num_workers = 0  # sanity: avoid worker forks
    cfg.pin_memory = False

    loader = build_coco_loader(cfg, split="eval", seed=seed)
    images, targets = next(iter(loader))

    B, C, H, W = images.shape
    num_targets_per_image = [int(t["labels"].numel()) for t in targets]
    all_labels = torch.cat([t["labels"] for t in targets]) if num_targets_per_image and sum(num_targets_per_image) > 0 else torch.empty(0, dtype=torch.long)
    cat_idx_min = int(all_labels.min()) if all_labels.numel() > 0 else None
    cat_idx_max = int(all_labels.max()) if all_labels.numel() > 0 else None

    checks = {
        "images_is_tensor": isinstance(images, torch.Tensor),
        "images_4d": images.dim() == 4,
        "channels_3": C == 3,
        "batch_size_match": B == batch_size,
        "h_w_positive": H > 0 and W > 0,
        "targets_len_match": len(targets) == B,
        "cat_idx_in_range": (all_labels.numel() == 0) or (cat_idx_min >= 0 and cat_idx_max < NUM_CLASSES),
    }
    sanity_pass = all(checks.values())

    result = {
        "sanity_pass": sanity_pass,
        "batch_shape": [B, C, H, W],
        "num_targets_per_image": num_targets_per_image,
        "cat_idx_range": [cat_idx_min, cat_idx_max],
        "seed": seed,
        "split": split,
        "checks": checks,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sanity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="COCO loader 1-batch sanity.")
    ap.add_argument("--split", default="val", choices=["val"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    run_dir = ROOT / "runs" / f"code-skeleton-loaders-coco-{_now_tag()}"
    result = run(args.split, args.batch_size, args.seed, run_dir)
    print(f"sanity → {run_dir / 'sanity.json'}")
    print(json.dumps({k: result[k] for k in ("sanity_pass", "batch_shape", "num_targets_per_image", "cat_idx_range", "seed")}, indent=2))
    return 0 if result["sanity_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
