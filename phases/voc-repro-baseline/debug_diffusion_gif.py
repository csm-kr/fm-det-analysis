"""DDIM 단계별 박스 denoise + NMS GIF 생성 — 10 장 (이미지 별).

각 GIF 의 프레임 구성 (총 10 프레임):
  frame 0       : init bboxes (random Gaussian)        — t = T-1
  frame 1..8    : DDIM step 1..8 의 head 출력 박스       — t 감소
  frame 9       : NMS 후 살아남은 박스 (강조)            — 최종 prediction

색 규약:
  lime  : GT
  red (alpha 80, thin)   : 해당 step 의 score >= 0.1 의 top boxes (raw)
  red (alpha 255, thick) : NMS 후 살아남은 박스 (마지막 프레임)
  하단 label             : 현재 step / DDIM t / 박스 수 / 평균 score

사용:
  TORCH_HOME=/workspace/fm-det/.cache/torch PYTHONPATH=/workspace/fm-det \
    python phases/voc-repro-baseline/debug_diffusion_gif.py \
      --run_dir runs/20260523-1416-voc-repro-baseline \
      --num_images 10 --ckpt last.pt --n_steps 9
산출:
  runs/{run_dir}/debug_out/diffusion_{idx:02d}_{voc_id}.gif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
from torchvision.ops import batched_nms

from datasets.voc.dataset import build_voc_loader
from models import build_diffusiondet
from utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
]


def _to_pil(img_chw: torch.Tensor, mean, std) -> Image.Image:
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    img = (img_chw.detach().cpu() * s + m).clamp(0, 1)
    return Image.fromarray((img.permute(1, 2, 0).numpy() * 255).astype("uint8"))


def _draw_frame(base_pil: Image.Image, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
                boxes: torch.Tensor, scores: torch.Tensor, classes: torch.Tensor,
                label: str, nms_mode: bool = False, init_mode: bool = False) -> Image.Image:
    """한 프레임.

    - GT: lime, width=3
    - init_mode: 모든 박스 score 0 → 균일 흐린 빨강 (alpha=40, width=1) — 수렴 시작점 보여주기
    - nms_mode: 두꺼운 빨강 (alpha=255, width=3) + class:score 라벨
    - 일반 ddim step: alpha 와 width 가 score 에 비례 → 낮은 score 흐림, 높은 score 진함.
      이렇게 해야 500 박스가 점점 객체로 수렴하는 denoise 패턴이 보임 (score thresh 로 박스가
      사라지는 게 아니라, 흐려졌다 진해지는 형태).
    """
    pil = base_pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    # GT lime — 먼저 깔고 (semi-transparent 로 ghost ref)
    for j in range(gt_boxes.shape[0]):
        x1, y1, x2, y2 = gt_boxes[j].tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 200), width=2)
        ln = int(gt_labels[j])
        name = VOC_CLASSES[ln] if ln < len(VOC_CLASSES) else str(ln)
        draw.text((x1 + 2, max(y1 - 12, 0)), f"GT:{name}", fill=(0, 255, 0, 255))
    # pred boxes (GT 위에 덮어 — 좋은 prediction 일수록 GT 와 겹쳐 잘 안 보였던 문제 해결)
    for j in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[j].tolist()
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(pil.size[0] - 1.0, x2); y2 = min(pil.size[1] - 1.0, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if nms_mode:
            width = 4
            alpha = 255
        elif init_mode:
            width = 1
            alpha = 60
        else:
            sc = float(scores[j])
            # 흐릿 → 진함 linear: score 0 → alpha 70, score 1 → alpha 255 (base 올림)
            alpha = int(70 + min(max(sc, 0.0), 1.0) * 185)
            width = 1 if sc < 0.3 else (2 if sc < 0.6 else 3)
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, alpha), width=width)
        if nms_mode:
            cn = int(classes[j])
            name = VOC_CLASSES[cn] if cn < len(VOC_CLASSES) else str(cn)
            sc = float(scores[j])
            draw.text((x1 + 2, min(y2 + 2, pil.size[1] - 12)),
                       f"{name}:{sc:.2f}", fill=(255, 255, 0, 255))
    # label bar (좌상단)
    draw.rectangle([0, 0, pil.size[0], 18], fill=(0, 0, 0, 180))
    draw.text((4, 2), label, fill=(255, 255, 255, 255))
    out = Image.alpha_composite(pil, overlay)
    return out.convert("RGB")


@torch.no_grad()
def run_one(model, image: torch.Tensor, n_steps: int, device, amp_dtype,
            top_per_step: int = 200, nms_thresh: float = 0.5, max_nms: int = 100):
    """1 이미지 → DDIM 단계별 (boxes, scores, classes) 리스트 + 최종 NMS 결과."""
    sampler = model.sampler
    decoder = model.decoder
    backbone = model.backbone

    images = image.unsqueeze(0).to(device)
    B = 1
    H, W = images.shape[2:]

    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
        features = backbone(images)

    # init
    init_cxcywh = sampler.sample_infer_init_boxes(B, model.num_proposals, device)
    cxcywh01 = (init_cxcywh / sampler.signal_scale + 1.0) * 0.5
    cxcywh01 = cxcywh01.clamp(0.0, 1.0)
    xyxy01 = box_cxcywh_to_xyxy(cxcywh01)
    img_size_t = torch.tensor([W, H, W, H], device=device).float()
    xyxy_img = xyxy01 * img_size_t

    T = sampler.timesteps
    ts = torch.linspace(-1, T - 1, n_steps + 1).long().to(device)
    ts = ts.flip(0)

    frames_data = []  # list of (boxes, scores, classes, t_value, kind)
    # frame 0 — init (no pred yet, just init xyxy_img, 모든 박스 = init)
    frames_data.append({
        "boxes": xyxy_img[0].cpu(),
        "scores": torch.zeros(xyxy_img.shape[1]),  # 0 score (필터링 X)
        "classes": torch.zeros(xyxy_img.shape[1], dtype=torch.long),
        "t": int(ts[0].item()),
        "kind": "init",
    })

    x_t = init_cxcywh
    bboxes = xyxy_img
    last_logits, last_xyxy = None, None
    for i in range(n_steps):
        t_cur = ts[i].repeat(B).clamp(min=0)
        t_next = ts[i + 1].repeat(B).clamp(min=0)
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            pred_logits, pred_xyxy_img = decoder(features, bboxes, t_cur, is_eval=True)
        pred_logits = pred_logits[:, 0].float()    # [B, N, C]
        pred_xyxy_img = pred_xyxy_img[:, 0].float()  # [B, N, 4]
        last_logits, last_xyxy = pred_logits, pred_xyxy_img

        # 시각화: top_per_step boxes (score 필터 없이 alpha 만으로 표현)
        sc = pred_logits[0].sigmoid()
        flat_scs = sc.reshape(-1)
        N_b, C = sc.shape
        flat_box = pred_xyxy_img[0].unsqueeze(1).expand(N_b, C, 4).reshape(-1, 4)
        flat_cls = torch.arange(C, device=device).unsqueeze(0).expand(N_b, C).reshape(-1)
        # 항상 top-K 만 (대부분 동일한 박스의 다른 class 들이라 N*C 다 그리면 중복)
        k = min(top_per_step, flat_scs.numel())
        top_s, top_i = flat_scs.topk(k)
        frames_data.append({
            "boxes": flat_box[top_i].cpu(),
            "scores": top_s.cpu(),
            "classes": flat_cls[top_i].cpu(),
            "t": int(t_cur.item()),
            "kind": "ddim",
        })

        if i < n_steps - 1:
            pred_xyxy01 = pred_xyxy_img / img_size_t
            pred_xyxy01 = pred_xyxy01.clamp(0.0, 1.0)
            pred_cxcywh01 = box_xyxy_to_cxcywh(pred_xyxy01)
            pred_x0 = (pred_cxcywh01 * 2.0 - 1.0) * sampler.signal_scale
            pred_x0 = pred_x0.clamp(-sampler.signal_scale, sampler.signal_scale)
            x_t = sampler.ddim_step(x_t, t_cur, t_next, pred_x0)
            cxcywh01 = (x_t / sampler.signal_scale + 1.0) * 0.5
            cxcywh01 = cxcywh01.clamp(0.0, 1.0)
            xyxy01 = box_cxcywh_to_xyxy(cxcywh01)
            bboxes = xyxy01 * img_size_t

    # final NMS frame
    sc = last_logits[0].sigmoid()
    N_b, C = sc.shape
    flat_box = last_xyxy[0].unsqueeze(1).expand(N_b, C, 4).reshape(-1, 4)
    flat_scs = sc.reshape(-1)
    flat_cls = torch.arange(C, device=device).unsqueeze(0).expand(N_b, C).reshape(-1)
    m = flat_scs >= 0.05
    f_b = flat_box[m]; f_s = flat_scs[m]; f_c = flat_cls[m]
    if f_b.numel() > 0:
        keep = batched_nms(f_b, f_s, f_c, iou_threshold=nms_thresh)
        keep = keep[:max_nms]
        # viz 용도라 score 0.3 이상만
        kb, ks, kc = f_b[keep].cpu(), f_s[keep].cpu(), f_c[keep].cpu()
        viz_mask = ks >= 0.3
        frames_data.append({
            "boxes": kb[viz_mask],
            "scores": ks[viz_mask],
            "classes": kc[viz_mask],
            "t": 0,
            "kind": "nms",
        })
    else:
        frames_data.append({"boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
                            "classes": torch.zeros(0, dtype=torch.long), "t": 0, "kind": "nms"})

    return frames_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--num_images", type=int, default=10)
    ap.add_argument("--n_steps", type=int, default=9,
                     help="DDIM inference steps — frame 수 = n_steps + 2 (init + n_steps + NMS)")
    ap.add_argument("--ckpt", default="last.pt", choices=["last.pt", "best.pt"])
    ap.add_argument("--frame_ms", type=int, default=600, help="GIF frame duration (ms)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "debug_out"
    out_dir.mkdir(exist_ok=True)

    cfg = OmegaConf.load(run_dir / "config.yaml")
    cfg.data.batch_size = 1
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    loader = build_voc_loader(cfg.data, split="eval", seed=int(cfg.seed))

    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)
    state = torch.load(run_dir / "checkpoints" / args.ckpt, map_location=device,
                        weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"loaded {args.ckpt} (epoch={state.get('epoch')} iter={state.get('iter')})  "
          f"device={device}  n_steps={args.n_steps}")

    for idx, (images, targets) in enumerate(loader):
        if idx >= args.num_images:
            break
        tgt = targets[0]
        gt_boxes = tgt["boxes"].cpu().float()
        gt_labels = tgt["labels"].cpu().long()

        frames_data = run_one(model, images[0], n_steps=args.n_steps,
                               device=device, amp_dtype=amp_dtype)

        base_pil = _to_pil(images[0], list(cfg.data.mean), list(cfg.data.std))
        frames = []
        for fi, fd in enumerate(frames_data):
            init = (fd["kind"] == "init")
            nms = (fd["kind"] == "nms")
            if init:
                label = f"step 0/{args.n_steps}  INIT random  t={fd['t']:>4d}  N_boxes={fd['boxes'].shape[0]} (faint)"
            elif nms:
                label = f"FINAL + NMS  iou=0.5  N_kept={fd['boxes'].shape[0]} (score>=0.3)"
            else:
                step_i = fi  # 1-based index of ddim
                if fd['scores'].numel():
                    label = (f"step {step_i}/{args.n_steps}  DDIM denoise  t={fd['t']:>4d}  "
                              f"top{fd['boxes'].shape[0]} (alpha=score)  "
                              f"max_score={float(fd['scores'].max()):.2f}")
                else:
                    label = f"step {step_i}/{args.n_steps}  DDIM denoise  t={fd['t']:>4d}  empty"
            frame = _draw_frame(base_pil, gt_boxes, gt_labels,
                                  fd["boxes"], fd["scores"], fd["classes"],
                                  label=label, nms_mode=nms, init_mode=init)
            frames.append(frame)

        out_path = out_dir / f"diffusion_{idx:02d}_{tgt['voc_id'].replace('/', '_')}.gif"
        frames[0].save(str(out_path), save_all=True, append_images=frames[1:],
                        duration=args.frame_ms, loop=0)
        print(f"[{idx:02d}] {tgt['voc_id']}  → {out_path.name}  "
              f"frames={len(frames)}  size={base_pil.size}")


if __name__ == "__main__":
    main()
