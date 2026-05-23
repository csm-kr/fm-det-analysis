# Step 0: transforms-common

## 읽어야 할 파일
- `/CLAUDE.md` (detectron2 금지)
- `/DiffusionDet/src/data/dataset.py` (참조 only — transforms 정책)

## 작업
`datasets/transforms.py` 신설 — COCO/VOC 공통 transforms.

시그니처:
```python
def build_transforms(short_sides, max_size, flip_prob, mean, std, is_train):
    """List[Callable[(PIL.Image, dict), (PIL.Image, dict)]] 반환."""
```

요소:
- `RandomResize(short_sides, max_size)` — train: random pick / eval: short_sides[-1].
- `RandomHorizontalFlip(prob)` — image flip + boxes flip (xyxy: x_new = W - x).
- `ToTensor()` — PIL → tensor [3,H,W] (0-1).
- `Normalize(mean, std)` — ImageNet mean/std.

각 transform 은 `(image, target) -> (image, target)`. target dict = `{"boxes": Tensor[N,4] xyxy, "labels": Tensor[N], "image_id": int, "orig_size": (H,W)}`.

## AC
```bash
python3 -c "
from PIL import Image
import torch
from datasets.transforms import build_transforms
t = build_transforms([800], 1333, 0.5, [0.485,0.456,0.406], [0.229,0.224,0.225], is_train=True)
img = Image.new('RGB', (640, 480))
tgt = {'boxes': torch.tensor([[10.,20.,100.,200.]]), 'labels': torch.tensor([0]), 'image_id': 0, 'orig_size': (480,640)}
img2, tgt2 = t(img, tgt)
assert img2.shape[0] == 3, img2.shape
assert tgt2['boxes'].shape == (1,4)
print('OK')
"
```

## 금지사항
- detectron2.data.transforms 사용 금지.
- numpy 만으로 충분하면 torchvision.transforms 도 사용 가능 (정합성 위해 torchvision 권장).
