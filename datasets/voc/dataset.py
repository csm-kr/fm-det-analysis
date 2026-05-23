"""PASCAL VOC 2007/2012 detection PyTorch Dataset + DataLoader.

XML 파싱 (xml.etree, stdlib). detectron2 금지.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from datasets.transforms import build_transforms, collate_fn
from datasets.voc.sanity import SPLIT_MAP, VOC_CLASSES, _parse_xml


def _list_ids(devkit: Path, year: str, split: str) -> list[str]:
    f = devkit / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    return [line.strip() for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]


class VOCDetection(Dataset):
    """VOC detection dataset.

    split ∈ {voc07-trainval, voc07-test, voc12-trainval, voc-trainval-combined}.
    학습 분할은 `difficult=1` 박스 제외 (VOC 공식 컨벤션). 평가는 별도.
    """

    CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}

    def __init__(self, devkit_root: str | Path, split: str, transforms,
                 drop_difficult: bool = True, drop_empty: bool = True):
        self.devkit = Path(devkit_root)
        self.transforms = transforms
        self.drop_difficult = drop_difficult

        if split == "voc-trainval-combined":
            ids = [("2007", i) for i in _list_ids(self.devkit, "2007", "trainval")]
            ids += [("2012", i) for i in _list_ids(self.devkit, "2012", "trainval")]
        else:
            year, sp = SPLIT_MAP[split]
            ids = [(year, i) for i in _list_ids(self.devkit, year, sp)]

        if drop_empty:
            ids = [(y, i) for y, i in ids if _parse_xml(self.devkit / f"VOC{y}" / "Annotations" / f"{i}.xml")["boxes"]]
        self.entries = ids

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        year, img_id = self.entries[idx]
        voc_year = self.devkit / f"VOC{year}"
        img_path = voc_year / "JPEGImages" / f"{img_id}.jpg"
        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        info = _parse_xml(voc_year / "Annotations" / f"{img_id}.xml")
        boxes, labels, difficult = [], [], []
        for bx in info["boxes"]:
            if self.drop_difficult and bx["difficult"]:
                continue
            if bx["name"] not in self.CLASS_TO_IDX:
                continue
            boxes.append([bx["xmin"], bx["ymin"], bx["xmax"], bx["ymax"]])
            labels.append(self.CLASS_TO_IDX[bx["name"]])
            difficult.append(bx["difficult"])

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.long)
        difficult_t = torch.as_tensor(difficult, dtype=torch.long)

        # image_id: VOC ID 는 string — int 변환 (year*1e7 + id)
        try:
            iid = int(year) * 10_000_000 + int(img_id)
        except ValueError:
            iid = hash(f"{year}-{img_id}") & 0xFFFFFFF

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "difficult": difficult_t,
            "image_id": iid,
            "voc_id": f"{year}/{img_id}",
            "orig_size": (H, W),
            "size": (H, W),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


def build_voc_loader(cfg, split: str, seed: int) -> DataLoader:
    """split ∈ {'train', 'eval'} (cfg.train_split / cfg.eval_split 로 매핑)."""
    is_train = (split == "train")
    sp_name = cfg.train_split if is_train else cfg.eval_split
    tcfg = cfg.transforms.train if is_train else cfg.transforms.eval
    transforms = build_transforms(
        short_sides=list(tcfg.short_sides),
        max_size=tcfg.max_size,
        flip_prob=tcfg.flip_prob,
        mean=list(cfg.mean), std=list(cfg.std),
        is_train=is_train,
    )
    ds = VOCDetection(cfg.devkit_root, sp_name, transforms,
                      drop_difficult=is_train, drop_empty=is_train)

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=is_train,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=collate_fn, drop_last=is_train, generator=generator,
    )
