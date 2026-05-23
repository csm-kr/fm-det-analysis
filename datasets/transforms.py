"""COCO/VOC 공통 detection transforms.

DiffusionDet 동치: short side ∈ short_sides (random pick) / max_size cap + flip 0.5.
각 transform 은 (PIL.Image | Tensor, target_dict) → (..., target_dict).
target_dict = {boxes: Tensor[N,4] xyxy, labels: Tensor[N], image_id: int, orig_size: (H,W), ...}
"""

from __future__ import annotations

import random
from typing import Sequence

import torch
import torchvision.transforms.functional as TF
from PIL import Image


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    """short side ∈ short_sides (random) / max_size cap. boxes 도 같이 scale."""

    def __init__(self, short_sides: Sequence[int], max_size: int):
        self.short_sides = list(short_sides)
        self.max_size = max_size

    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            _, h, w = image.shape
        short = random.choice(self.short_sides) if len(self.short_sides) > 1 else self.short_sides[0]
        scale = short / min(h, w)
        if max(h, w) * scale > self.max_size:
            scale = self.max_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        image = TF.resize(image, [new_h, new_w], antialias=True)
        if "boxes" in target and len(target["boxes"]) > 0:
            ratio_w, ratio_h = new_w / w, new_h / h
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] *= ratio_w
            boxes[:, [1, 3]] *= ratio_h
            target["boxes"] = boxes
        target["size"] = (new_h, new_w)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = TF.hflip(image)
            if isinstance(image, Image.Image):
                w, _ = image.size
            else:
                _, _, w = image.shape
            if "boxes" in target and len(target["boxes"]) > 0:
                boxes = target["boxes"].clone()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target


class ToTensor:
    def __call__(self, image, target):
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        return image, target


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        self.mean = list(mean)
        self.std = list(std)

    def __call__(self, image, target):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, target


def build_transforms(short_sides: Sequence[int], max_size: int, flip_prob: float,
                     mean: Sequence[float], std: Sequence[float], is_train: bool):
    ts: list = [RandomResize(short_sides, max_size)]
    if is_train and flip_prob > 0:
        ts.append(RandomHorizontalFlip(flip_prob))
    ts.append(ToTensor())
    ts.append(Normalize(mean, std))
    return Compose(ts)


def collate_fn(batch):
    """가변 크기 이미지 → max H,W zero-padding. (images: Tensor[B,3,H,W], targets: list[dict])."""
    images, targets = list(zip(*batch))
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # pad to 32 multiple (FPN p5 = /32)
    max_h = ((max_h + 31) // 32) * 32
    max_w = ((max_w + 31) // 32) * 32
    out = torch.zeros(len(images), 3, max_h, max_w, dtype=images[0].dtype)
    masks = torch.ones(len(images), max_h, max_w, dtype=torch.bool)
    for i, img in enumerate(images):
        c, h, w = img.shape
        out[i, :, :h, :w] = img
        masks[i, :h, :w] = False
    for tgt, mask in zip(targets, masks):
        tgt["padding_mask"] = mask
    return out, list(targets)
