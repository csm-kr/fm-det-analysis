# Step 2: voc-dataset-loader

## 작업
`datasets/voc/dataset.py` 신설.

```python
class VOCDetection(torch.utils.data.Dataset):
    def __init__(self, devkit_root, year, split, transforms): ...

def build_voc_loader(cfg, split) -> DataLoader: ...  # split ∈ {voc07-trainval, voc07-test, voc12-trainval, voc-trainval-combined}
```

20 클래스 → 0-19 인덱싱. `difficult=1` 박스는 학습 set 에서 제외 (VOC 공식 컨벤션 — eval 에서는 별도 처리).

## AC
```bash
python -m datasets.voc.sanity_loader --split voc07-trainval --batch-size 2 --seed 42

jq -e '.sanity_pass == true' runs/code-skeleton-loaders-voc-*/sanity.json
```

## 금지사항
- detectron2 / pycocotools 사용 금지 (XML 파싱은 datasets/voc/sanity.py 의 _parse_xml 재사용).
- difficult=1 박스 학습에 포함 금지.
