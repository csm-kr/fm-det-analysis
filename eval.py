"""DiffusionDet 평가 진입점 — Hydra @main.

사용:
    python eval.py +experiment=coco-repro-baseline seed=42 run_dir=runs/{id}

산출: runs/{run_dir}/eval-{HHmm}/eval.json — { metric_primary: <mAP>, ... }
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from datasets.coco.dataset import build_coco_loader
from datasets.voc.dataset import build_voc_loader
from evals import coco_eval, voc_eval
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
    if str(cfg.run_dir) == "???":
        raise RuntimeError("run_dir required. e.g. run_dir=runs/20260524-0100-baseline")

    seed = int(cfg.seed)
    _set_seed(seed)

    device = torch.device(cfg.device)

    if cfg.data.name == "coco":
        loader = build_coco_loader(cfg.data, "eval", seed)
        run_eval = lambda m: coco_eval(m, loader, device)
    elif cfg.data.name == "voc":
        loader = build_voc_loader(cfg.data, "eval", seed)
        run_eval = lambda m: voc_eval(m, loader, device, num_classes=cfg.data.num_classes)
    else:
        raise ValueError(f"unknown data.name = {cfg.data.name}")

    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)

    ckpt_path = Path(cfg.ckpt)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"] if "model" in state else state, strict=False)
        print(f"loaded ckpt {ckpt_path}")
    else:
        print(f"WARN: ckpt not found at {ckpt_path} — running with random init")

    metrics = run_eval(model)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"-> {out_dir}/eval.json")


if __name__ == "__main__":
    main()
