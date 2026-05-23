"""DiffusionDet 학습 진입점 — Hydra @main.

사용:
    python train.py +experiment=coco-repro-baseline seed=42
    python train.py +experiment=voc-repro-baseline  seed=42

dry-run (1 iter):
    python train.py +experiment=coco-repro-baseline seed=42 train.max_iters=1

산출: runs/{ts}-{tag}/{config.yaml, git_rev.txt, seed.txt, metrics.csv, checkpoints/}
"""

from __future__ import annotations

import csv
import os
import random
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from datasets.coco.dataset import build_coco_loader
from datasets.voc.dataset import build_voc_loader
from losses import build_criterion
from models import build_diffusiondet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loaders(cfg, seed: int):
    if cfg.data.name == "coco":
        train_loader = build_coco_loader(cfg.data, "train", seed)
        eval_loader = build_coco_loader(cfg.data, "eval", seed)
    elif cfg.data.name == "voc":
        train_loader = build_voc_loader(cfg.data, "train", seed)
        eval_loader = build_voc_loader(cfg.data, "eval", seed)
    else:
        raise ValueError(f"unknown data.name = {cfg.data.name}")
    return train_loader, eval_loader


def _targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {**t, "boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
        for t in targets
    ]


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.seed is None or str(cfg.seed) == "???":
        raise RuntimeError("seed required (CLAUDE.md CRITICAL). e.g. seed=42")

    seed = int(cfg.seed)
    _set_seed(seed)

    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    (out_dir / "seed.txt").write_text(f"{seed}\n")
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        rev = "unknown"
    (out_dir / "git_rev.txt").write_text(rev + "\n")

    device = torch.device(cfg.device)
    train_loader, eval_loader = _build_loaders(cfg, seed)

    # model.num_classes 가 ${data.num_classes} interpolation 일 수 있어 force-set
    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)
    criterion = build_criterion(cfg.data.num_classes, cfg.loss).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = cfg.train.optimizer
    optim = torch.optim.AdamW(trainable, lr=float(optim_cfg.lr),
                              weight_decay=float(optim_cfg.weight_decay))

    sch_cfg = cfg.train.scheduler
    multistep = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=list(sch_cfg.milestones), gamma=float(sch_cfg.gamma),
    )
    warmup_iters = int(sch_cfg.get("warmup_iters", 0))
    warmup_factor = float(sch_cfg.get("warmup_factor", 1.0))

    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "iter", "loss_total", "loss_cls", "loss_l1", "loss_giou",
            "grad_norm", "lr", "elapsed_sec",
        ])
    tb_writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    grad_clip = float(cfg.train.grad_clip)
    log_interval = int(cfg.train.log_interval)
    epochs = int(cfg.train.epochs)
    max_iters = int(cfg.train.get("max_iters", 0))  # 0 = unlimited (정식 학습)

    iter_count = 0
    t_start = time.monotonic()
    last_loss = None
    finished = False

    for epoch in range(epochs):
        if finished:
            break
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, targets = batch
            images = images.to(device, non_blocking=True)
            targets = _targets_to_device(targets, device)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss_total"]
            if not torch.isfinite(loss).all():
                # fp16 overflow 등 — scaler 가 처리 못 한 경우. step skip + 경고.
                print(f"WARN: non-finite loss at epoch {epoch} iter {iter_count} (skipping)")
                optim.zero_grad(set_to_none=True)
                continue

            # LR warmup (iter ≤ warmup_iters): linear from warmup_factor → 1.0
            if warmup_iters > 0 and iter_count < warmup_iters:
                alpha = iter_count / max(warmup_iters, 1)
                w = warmup_factor + (1.0 - warmup_factor) * alpha
                for pg in optim.param_groups:
                    pg["lr"] = float(optim_cfg.lr) * w

            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
            scaler.step(optim)
            scaler.update()

            iter_count += 1
            last_loss = float(loss.detach())
            cur_lr = optim.param_groups[0]["lr"]
            cls_v = float(loss_dict["loss_cls"])
            l1_v = float(loss_dict["loss_l1"])
            giou_v = float(loss_dict["loss_giou"])
            gnorm = float(grad_norm)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, iter_count, last_loss, cls_v, l1_v, giou_v,
                    gnorm, cur_lr, round(time.monotonic() - t_start, 2),
                ])
            tb_writer.add_scalar("train/loss_total", last_loss, iter_count)
            tb_writer.add_scalar("train/loss_cls", cls_v, iter_count)
            tb_writer.add_scalar("train/loss_l1", l1_v, iter_count)
            tb_writer.add_scalar("train/loss_giou", giou_v, iter_count)
            # AMP 첫 step grad_norm NaN 은 skip (TB 가 NaN 처리 못 함)
            if gnorm == gnorm:
                tb_writer.add_scalar("train/grad_norm", gnorm, iter_count)
            tb_writer.add_scalar("train/lr", cur_lr, iter_count)
            if iter_count % log_interval == 0 or iter_count == 1:
                print(f"[epoch {epoch} iter {iter_count}] loss={last_loss:.4f} "
                      f"grad_norm={gnorm:.2f} lr={cur_lr:.2e}")
            # epoch 내에서도 1000 iter 마다 last.pt 저장 (crash 시 손실 최소화)
            if iter_count % 1000 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": multistep.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "iter": iter_count,
                }, ckpt_dir / "last.pt")
            if max_iters > 0 and iter_count >= max_iters:
                finished = True
                break
        multistep.step()
        # ckpt — last.pt 매 epoch
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": multistep.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "iter": iter_count,
        }, ckpt_dir / "last.pt")

    tb_writer.close()
    print(f"Done. epoch={epoch} iter={iter_count} last_loss={last_loss} runs/{out_dir.name}")


if __name__ == "__main__":
    main()
