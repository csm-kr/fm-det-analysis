"""model-sanity-overfit — 1 batch overfit + shape/grad/param check.

harness §2-2 패턴. 산출: runs/model-sanity-{ts}/{sanity.json, report.md}
실행:
    python -m models.sanity --device cuda --seed 42 --steps 100 --batch-size 2
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
from omegaconf import OmegaConf

from losses import build_criterion
from models import build_diffusiondet


def _build_model_cfg(num_classes: int):
    raw = OmegaConf.load("configs/model/diffusiondet.yaml")
    raw.num_classes = num_classes
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="runs/model-sanity-{ts}")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    kst = timezone(timedelta(hours=9))
    ts = datetime.now(kst).strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out.format(ts=ts))
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_model_cfg(args.num_classes)
    loss_cfg = OmegaConf.load("configs/loss/diffusion.yaml")

    device = torch.device(args.device)
    model = build_diffusiondet(cfg).to(device)
    criterion = build_criterion(args.num_classes, loss_cfg).to(device)
    model.train()

    B, H, W = args.batch_size, args.height, args.width
    images = torch.randn(B, 3, H, W, device=device)
    targets = []
    rng = torch.Generator(device=device).manual_seed(args.seed)
    for _ in range(B):
        n = 2
        xy0 = torch.rand(n, 2, device=device, generator=rng) * 0.5
        wh = torch.rand(n, 2, device=device, generator=rng) * 0.4 + 0.1
        xy1 = (xy0 + wh).clamp(max=1.0)
        boxes01 = torch.cat([xy0, xy1], dim=1)
        boxes = boxes01 * torch.tensor([W, H, W, H], device=device)
        labels = torch.randint(0, args.num_classes, (n,), device=device, generator=rng)
        targets.append({"boxes": boxes, "labels": labels})

    trainable = [p for p in model.parameters() if p.requires_grad]
    param_count_m = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_count_m = sum(p.numel() for p in trainable) / 1e6

    out_train = model(images, targets)
    K = cfg.num_heads
    forward_shape_ok = (
        tuple(out_train["pred_logits"].shape) == (B, K, cfg.num_proposals, args.num_classes)
        and tuple(out_train["pred_boxes"].shape) == (B, K, cfg.num_proposals, 4)
    )
    pred_logits_shape = list(out_train["pred_logits"].shape)
    pred_boxes_shape = list(out_train["pred_boxes"].shape)

    optim = torch.optim.AdamW(trainable, lr=args.lr)
    losses: list[float] = []
    all_params_have_grad = False
    params_with_grad = 0
    params_total = len(trainable)

    for step in range(args.steps):
        outputs = model(images, targets)
        loss = criterion(outputs, targets)["loss_total"]
        assert torch.isfinite(loss).all(), f"non-finite loss at step {step}: {loss}"
        optim.zero_grad()
        loss.backward()
        if step == 0:
            params_with_grad = sum(1 for p in trainable if p.grad is not None)
            all_params_have_grad = params_with_grad == params_total
        optim.step()
        losses.append(float(loss.detach()))

    first_loss = losses[0]
    last_loss = losses[-1]
    # DiffusionDet set loss (focal×2 + L1×5 + GIoU×2) absolute 값이 큼.
    # 학습이 동작한다는 신호는 *상대 감소* — last < first × 0.8 (20%+ drop).
    loss_decreased = last_loss < first_loss * 0.8
    loss_drop_ratio = round(1.0 - (last_loss / first_loss), 4) if first_loss > 0 else 0.0

    sanity = {
        "forward_shape_ok": bool(forward_shape_ok),
        "all_params_have_grad": bool(all_params_have_grad),
        "params_with_grad": int(params_with_grad),
        "params_total": int(params_total),
        "param_count_m": round(param_count_m, 2),
        "trainable_count_m": round(trainable_count_m, 2),
        "loss_decreased": bool(loss_decreased),
        "loss_drop_ratio": loss_drop_ratio,
        "overfit_one_batch_loss": round(last_loss, 4),
        "first_loss": round(first_loss, 4),
        "last_loss": round(last_loss, 4),
        "losses_first10": [round(x, 4) for x in losses[:10]],
        "losses_last10": [round(x, 4) for x in losses[-10:]],
        "pred_logits_shape": pred_logits_shape,
        "pred_boxes_shape": pred_boxes_shape,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": B,
        "image_hw": [H, W],
    }
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2))

    report = (
        "# Model Sanity Report (model-sanity-overfit)\n\n"
        f"- Generated: {ts}\n- Device: {device}\n- Seed: {args.seed}\n"
        f"- Steps: {args.steps}  Batch: {B}  Image: {H}x{W}\n\n"
        "| 항목 | 값 |\n|------|----|\n"
        f"| forward_shape_ok | {sanity['forward_shape_ok']} |\n"
        f"| pred_logits.shape | {sanity['pred_logits_shape']} |\n"
        f"| pred_boxes.shape | {sanity['pred_boxes_shape']} |\n"
        f"| all_params_have_grad | {sanity['all_params_have_grad']} "
        f"({sanity['params_with_grad']}/{sanity['params_total']}) |\n"
        f"| param_count_m | {sanity['param_count_m']} M |\n"
        f"| trainable_count_m | {sanity['trainable_count_m']} M |\n"
        f"| first_loss | {sanity['first_loss']} |\n"
        f"| last_loss | {sanity['last_loss']} |\n"
        f"| loss_drop_ratio | {sanity['loss_drop_ratio']} ({sanity['loss_decreased']}) |\n\n"
        f"## Loss trace\n- first 10: {sanity['losses_first10']}\n"
        f"- last 10:  {sanity['losses_last10']}\n"
    )
    (out_dir / "report.md").write_text(report)

    print(json.dumps(sanity, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
