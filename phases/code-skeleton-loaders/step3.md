# Step 3: hydra-configs

## 작업
- `configs/train.yaml` — 학습 진입점 base (`defaults: [data: coco, model: diffusiondet, loss: diffusion, train: baseline]`).
- `configs/eval.yaml` — 평가 진입점.
- `configs/data/coco.yaml` — batch_size, num_workers, transforms 파라미터.
- `configs/data/voc.yaml` — VOC 버전.

핵심: **batch_size 결정 자리 = configs/data/coco.yaml** (재현성). GPU 96GB 라 batch=16 (DiffusionDet 동치) 가능 + 늘릴 여유.

## AC
```bash
test -f configs/data/coco.yaml && test -f configs/data/voc.yaml
python3 -c "from omegaconf import OmegaConf; cfg = OmegaConf.load('configs/data/coco.yaml'); assert cfg.batch_size == 16; print(cfg)"
```

## 금지사항
- batch_size 를 코드 안에 하드코딩 금지 — 항상 Hydra config.
