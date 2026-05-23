"""loss-sanity-50step — finite/grad-norm/loss-decreases 검증.

harness §2-3 패턴. 산출: runs/loss-sanity-{ts}/{sanity.json, report.md}
실행:
    python -m losses.sanity --device cuda --seed 42 --steps 50 --batch-size 2
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--out", default="runs/loss-sanity-{ts}")
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

    cfg = OmegaConf.load("configs/model/diffusiondet.yaml")
    cfg.num_classes = args.num_classes
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
    optim = torch.optim.AdamW(trainable, lr=args.lr)

    losses: list[float] = []
    per_layer_loss_records: list[dict] = []  # 50-step 마지막의 layer 0/last 기록
    grad_norms: list[float] = []
    nan_inf_count = 0

    for step in range(args.steps):
        out_t = model(images, targets)
        loss_dict = criterion(out_t, targets)
        loss = loss_dict["loss_total"]
        if not torch.isfinite(loss).all():
            nan_inf_count += 1
            (out_dir / "error.log").write_text(f"non-finite loss at step {step}: {loss}\n")
            break
        optim.zero_grad()
        loss.backward()
        # grad norm 측정 (clip 도 같이)
        total_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.grad_clip)
        gnorm = float(total_norm.detach())
        if not (gnorm == gnorm):  # NaN check
            nan_inf_count += 1
        grad_norms.append(gnorm)
        optim.step()
        losses.append(float(loss.detach()))
        if step == args.steps - 1:
            per_layer_loss_records = [
                {k: float(v.detach()) if torch.is_tensor(v) else None
                 for k, v in pl.items() if torch.is_tensor(v)}
                for pl in loss_dict["per_layer"]
            ]

    grad_norm_max = max(grad_norms) if grad_norms else float("inf")
    grad_norm_mean = sum(grad_norms) / len(grad_norms) if grad_norms else float("inf")
    loss_decreases = bool(len(losses) >= 2 and losses[-1] < losses[0] * 0.8)

    sanity = {
        "nan_inf_count": int(nan_inf_count),
        "grad_norm_max": round(grad_norm_max, 4),
        "grad_norm_mean": round(grad_norm_mean, 4),
        "loss_decreases": bool(loss_decreases),
        "first_loss": round(losses[0], 4) if losses else None,
        "last_loss": round(losses[-1], 4) if losses else None,
        "loss_drop_ratio": round(1.0 - (losses[-1] / losses[0]), 4) if losses and losses[0] > 0 else None,
        "losses_first10": [round(x, 4) for x in losses[:10]],
        "losses_last10": [round(x, 4) for x in losses[-10:]],
        "grad_norms_first10": [round(x, 4) for x in grad_norms[:10]],
        "grad_norms_last10": [round(x, 4) for x in grad_norms[-10:]],
        "per_layer_loss_last_step": per_layer_loss_records,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "completed_steps": len(losses),
        "batch_size": B,
        "image_hw": [H, W],
        "grad_clip": args.grad_clip,
    }
    (out_dir / "sanity.json").write_text(json.dumps(sanity, indent=2))

    report = (
        "# Loss Sanity Report (loss-sanity-50step)\n\n"
        f"- Generated: {ts}\n- Device: {device}\n- Seed: {args.seed}\n"
        f"- Steps: {args.steps} (completed={sanity['completed_steps']})  Batch: {B}  Image: {H}x{W}\n"
        f"- grad_clip: {args.grad_clip}\n\n"
        "| 항목 | 값 |\n|------|----|\n"
        f"| nan_inf_count | {sanity['nan_inf_count']} |\n"
        f"| grad_norm_max | {sanity['grad_norm_max']} |\n"
        f"| grad_norm_mean | {sanity['grad_norm_mean']} |\n"
        f"| loss_decreases | {sanity['loss_decreases']} |\n"
        f"| first_loss | {sanity['first_loss']} |\n"
        f"| last_loss | {sanity['last_loss']} |\n"
        f"| loss_drop_ratio | {sanity['loss_drop_ratio']} |\n\n"
        f"## Loss trace\n- first 10: {sanity['losses_first10']}\n"
        f"- last 10:  {sanity['losses_last10']}\n\n"
        f"## Grad norm trace\n- first 10: {sanity['grad_norms_first10']}\n"
        f"- last 10:  {sanity['grad_norms_last10']}\n"
    )
    (out_dir / "report.md").write_text(report)

    print(json.dumps(sanity, indent=2))
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
