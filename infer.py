"""DiffusionDet 단일 이미지 추론 진입점 — Hydra @main.

사용:
    python infer.py +experiment=coco-repro-baseline seed=42 run_dir=runs/{id} image=path/to/img.jpg

산출: runs/{run_dir}/infer-{HHmm}/{image_basename}.{predictions.json, drawn.jpg}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw

from datasets.transforms import build_transforms
from models import build_diffusiondet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if cfg.seed is None or str(cfg.seed) == "???":
        raise RuntimeError("seed required. e.g. seed=42")
    image_path = cfg.get("image", None)
    if image_path is None:
        raise RuntimeError("image required. e.g. image=path/to/img.jpg")
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    seed = int(cfg.seed)
    _set_seed(seed)
    device = torch.device(cfg.device)

    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)
    ckpt_path = Path(cfg.ckpt)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
    model.eval()

    tcfg = cfg.data.transforms.eval
    transforms = build_transforms(
        short_sides=list(tcfg.short_sides), max_size=tcfg.max_size,
        flip_prob=tcfg.flip_prob,
        mean=list(cfg.data.mean), std=list(cfg.data.std),
        is_train=False,
    )

    pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil.size
    image_tensor, _ = transforms(pil, {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.long)})
    images = image_tensor.unsqueeze(0).to(device)
    cur_h, cur_w = image_tensor.shape[1:]
    sx, sy = orig_w / cur_w, orig_h / cur_h

    with torch.no_grad():
        out = model(images)
    scores = out["pred_logits"].sigmoid()[0]  # [N, C]
    boxes = out["pred_boxes"][0]              # [N, 4] xyxy in transformed-image px
    N, C = scores.shape
    score_thresh = float(cfg.get("score_thresh", 0.3))
    max_dets = int(cfg.get("max_dets", 100))

    flat = scores.reshape(-1)
    k = min(max_dets, flat.numel())
    top_scores, top_idx = flat.topk(k)
    box_idx = top_idx // C
    cls_idx = top_idx % C
    sel = boxes[box_idx].clone()
    sel[:, 0::2] *= sx
    sel[:, 1::2] *= sy

    detections = []
    for j in range(k):
        s = float(top_scores[j])
        if s < score_thresh:
            continue
        x1, y1, x2, y2 = sel[j].tolist()
        detections.append({"box_xyxy": [x1, y1, x2, y2], "score": s,
                           "class_idx": int(cls_idx[j])})

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = image_path.stem
    (out_dir / f"{base}.predictions.json").write_text(json.dumps(detections, indent=2))

    draw = ImageDraw.Draw(pil)
    for d in detections:
        x1, y1, x2, y2 = d["box_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(y1 - 10, 0)), f"{d['class_idx']}:{d['score']:.2f}", fill="red")
    pil.save(out_dir / f"{base}.drawn.jpg")
    print(f"-> {out_dir}/{base}.{{predictions.json,drawn.jpg}}  ({len(detections)} detections)")


if __name__ == "__main__":
    main()
