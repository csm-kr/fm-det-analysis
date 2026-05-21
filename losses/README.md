# losses/ — DiffusionDet SetCriterion

`detectron2` 없이 PyTorch + `torchvision.ops.generalized_box_iou` 만 사용.

## 전체 흐름 (gt ↔ pred → loss)

```mermaid
flowchart TB
    pred["model output<br/>pred_logits [B,K=6,N=500,C=80]<br/>pred_boxes [B,K,N,4] xyxy"]
    gt["targets (per image)<br/>boxes [M_i,4] xyxy<br/>labels [M_i]"]

    pred --> ploop["for k in 0..K-1<br/>(deep supervision)"]
    ploop --> layerk["layer k:<br/>pred_logits[:,k] [B,N,C]<br/>pred_boxes[:,k] [B,N,4]"]

    layerk --> matcher["SimOTAMatcher<br/>(matcher.py)"]
    gt --> matcher
    matcher --> idx["indices: list[B] of (pred_idx, tgt_idx)"]

    idx --> cls["focal loss<br/>(matched 1, others 0)"]
    layerk --> cls
    idx --> l1["L1 on matched<br/>(normalized cxcywh)"]
    layerk --> l1
    gt --> l1
    idx --> giou["1 - GIoU on matched<br/>(image coords xyxy)"]
    layerk --> giou
    gt --> giou

    cls --> sum["loss_cls × 2.0<br/>+ loss_l1 × 5.0<br/>+ loss_giou × 2.0"]
    l1 --> sum
    giou --> sum

    sum --> total["sum over K layers<br/>→ loss_total"]
```

## SimOTAMatcher 내부

```mermaid
flowchart LR
    p["pred [N,C], pred_boxes [N,4]"] --> cost
    g["gt boxes [M,4], gt labels [M]"] --> cost
    cost["cost = 2.0·focal + 5.0·L1(cxcywh norm) + 2.0·(1-GIoU)<br/>+ inf · (~center_in_box)"]

    g --> dk["dynamic_k<br/>= sum(top10 IoU per GT).clamp(min=1)"]
    p --> dk
    cost --> match["for each GT: top-k lowest cost"]
    dk --> match
    match --> conflict["충돌: 한 pred → 여러 GT?<br/>가장 낮은 cost GT 선택"]
    conflict --> out["indices: (pred_idx, tgt_idx)"]
```

## Loss 구성표

| 손실 | weight | normalize 기준 | 적용 범위 |
|------|--------|--------------|---------|
| focal (cls) | 2.0 | num_matched | **모든 N 박스** (positive=matched, negative=others) |
| L1 (bbox) | 5.0 | num_matched | **matched 만** (normalized cxcywh) |
| GIoU (bbox) | 2.0 | num_matched | **matched 만** (image coords xyxy) |
| 합산 | — | — | K=6 layer 모두 (deep supervision) → 12·(focal_k) + 30·(L1_k) + 12·(GIoU_k) |

## Matcher 핵심 차이 (Hungarian vs SimOTA)

| 항목 | Hungarian (DETR) | SimOTA (DiffusionDet, **본 구현**) |
|------|------------------|----------------------------------|
| GT ↔ pred 비율 | 1:1 | 1:k (k=dynamic) |
| 매칭 알고리즘 | linear_sum_assignment | top-k lowest cost per GT + conflict resolve |
| 매칭당 GT 수 | 1 | dynamic (top-10 IoU sum) |
| center prior | 없음 | center_in_box constraint (외부 → ∞ cost) |

## 검증
- 본 sanity: `loss_total = 38.5` finite, 모든 314 trainable param 에 `grad` 도달 (CPU 2-batch).
- GPU 50-step sanity 는 I-06 해소 후 (Blackwell sm_120 PyTorch 호환).
