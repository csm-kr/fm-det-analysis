"""COCO 2017 detection PyTorch Dataset + DataLoader.

pycocotools 만 사용 (detectron2 금지 — CLAUDE.md CRITICAL).
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import build_transforms, collate_fn


class CocoDetection(Dataset):
    """COCO instance detection dataset.

    Returns: (image: Tensor[3,H,W], target: dict).
    target = {boxes: Tensor[N,4] xyxy, labels: Tensor[N] in [0, num_classes),
              image_id: int, orig_size: (H,W), size: (H,W) after transform}.
    cat_id (1-90) → cat_idx (0-79) 매핑. background-only 이미지는 학습에서 제외.
    """

    def __init__(self, ann_file: str | Path, image_dir: str | Path,
                 transforms, drop_empty: bool = True):
        self.image_dir = Path(image_dir)
        self.coco = COCO(str(ann_file))
        self.transforms = transforms
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}
        ids = self.coco.getImgIds()
        if drop_empty:
            ids = [i for i in ids if self.coco.getAnnIds(imgIds=i)]
        self.image_ids = ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = self.image_dir / img_info["file_name"]
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        boxes = []
        labels = []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[a["category_id"]])
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.long)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": img_id,
            "orig_size": (H, W),
            "size": (H, W),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_coco_loader(cfg, split: str, seed: int) -> DataLoader:
    """split ∈ {'train', 'eval'}. cfg = configs/data/coco.yaml resolved."""
    is_train = (split == "train")
    tcfg = cfg.transforms.train if is_train else cfg.transforms.eval
    transforms = build_transforms(
        short_sides=list(tcfg.short_sides),
        max_size=tcfg.max_size,
        flip_prob=tcfg.flip_prob,
        mean=list(cfg.mean), std=list(cfg.std),
        is_train=is_train,
    )
    ann_file = cfg.ann_train if is_train else cfg.ann_eval
    image_dir = cfg.images_train if is_train else cfg.images_eval
    ds = CocoDetection(ann_file, image_dir, transforms, drop_empty=is_train)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=is_train,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate_fn, drop_last=is_train, generator=generator,
    )
