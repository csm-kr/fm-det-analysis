# Step 1: coco-dataset-loader

## 읽어야 할 파일
- `/DiffusionDet/src/data/dataset.py` (참조)
- `/docs/DATA_CARD.md` (COCO 80 cls)

## 작업
`datasets/coco/dataset.py` 신설.

```python
class CocoDetection(torch.utils.data.Dataset):
    def __init__(self, ann_file, image_dir, transforms): ...
    def __getitem__(self, idx) -> tuple[Tensor, dict]: ...
    def __len__(self) -> int: ...

def collate_fn(batch) -> tuple[Tensor, list[dict]]:
    """가변 크기 이미지를 max H,W 로 zero-padding. NestedTensor 대신 (images, list[target]) 반환."""

def build_coco_loader(cfg, split) -> DataLoader: ...
```

cat_id (1-90) → cat_idx (0-79) 매핑. background-only 이미지는 학습에서 제외 (DiffusionDet 동치).

## AC
```bash
python -m datasets.coco.sanity_loader --split val --batch-size 2 --seed 42
# runs/code-skeleton-loaders-coco-{ts}/sanity.json 작성

jq -e '.sanity_pass == true and .batch_shape != null' runs/code-skeleton-loaders-coco-*/sanity.json
```

sanity.json 스키마: `{sanity_pass: bool, batch_shape: [B,3,H,W], num_targets_per_image: [...], cat_idx_range: [min,max], seed: 42}`.

## 금지사항
- detectron2 사용 금지.
- test split 학습 사용 금지.
- background-only 이미지 학습 사용 금지 (DiffusionDet 동치).
