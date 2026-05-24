"""DiffusionDet 학습 진입점 — Hydra @main.

사용:
    python train.py +experiment=coco-repro-baseline seed=42
    python train.py +experiment=voc-repro-baseline  seed=42

dry-run (1 iter):
    python train.py +experiment=coco-repro-baseline seed=42 train.max_iters=1

산출: runs/{ts}-{tag}/{config.yaml, git_rev.txt, seed.txt, metrics.csv, checkpoints/}
"""

from __future__ import annotations

import csv
import json
import os
import random
import subprocess
import time
from pathlib import Path

import hydra
import numpy as np
import psutil
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_tensor

from datasets.coco.dataset import build_coco_loader
from datasets.voc.dataset import build_voc_loader
from evals import coco_eval, voc_eval
from losses import build_criterion
from models import build_diffusiondet


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_loaders(cfg, seed: int):
    if cfg.data.name == "coco":
        train_loader = build_coco_loader(cfg.data, "train", seed)
        eval_loader = build_coco_loader(cfg.data, "eval", seed)
    elif cfg.data.name == "voc":
        train_loader = build_voc_loader(cfg.data, "train", seed)
        eval_loader = build_voc_loader(cfg.data, "eval", seed)
    else:
        raise ValueError(f"unknown data.name = {cfg.data.name}")
    return train_loader, eval_loader


def _targets_to_device(targets: list[dict], device: torch.device) -> list[dict]:
    return [
        {**t, "boxes": t["boxes"].to(device), "labels": t["labels"].to(device)}
        for t in targets
    ]


def _print_startup(cfg, out_dir, rev, n_train, n_eval, device, amp_enabled, amp_dtype_str,
                   warmup_iters, warmup_factor, eval_interval):
    """학습 시작 시 핵심 설정 한 화면에 요약 — 다른 시점 재현 시 빠른 확인."""
    bar = "=" * 70
    optim_cfg = cfg.train.optimizer
    sch_cfg = cfg.train.scheduler
    lines = [
        bar,
        f"  fm-det training",
        bar,
        f"  data       : {cfg.data.name}  batch={cfg.data.batch_size}  num_classes={cfg.data.num_classes}",
        f"               train_imgs={n_train}  eval_imgs={n_eval}  iters/epoch={n_train // cfg.data.batch_size}",
        f"  model      : {cfg.model.name}  num_proposals={cfg.model.num_proposals}  heads={cfg.model.num_heads}",
        f"  loss       : {cfg.loss.name}  focal(α={cfg.loss.focal_alpha},γ={cfg.loss.focal_gamma})  "
        f"cls×{cfg.loss.class_weight}  l1×{cfg.loss.l1_weight}  giou×{cfg.loss.giou_weight}",
        f"  optimizer  : AdamW  lr={optim_cfg.lr}  weight_decay={optim_cfg.weight_decay}",
        f"  scheduler  : MultiStepLR  milestones={list(sch_cfg.milestones)}  gamma={sch_cfg.gamma}",
        f"  warmup     : {warmup_iters} iter × factor {warmup_factor}",
        f"  amp        : enabled={amp_enabled}  dtype={amp_dtype_str}  grad_clip={cfg.train.grad_clip}",
        f"  epochs     : {cfg.train.epochs}  log_interval={cfg.train.log_interval}  eval_interval_epoch={eval_interval}",
        f"  seed       : {cfg.seed}  device={device}  ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'})",
        f"  git rev    : {rev[:12]}",
        f"  out_dir    : {out_dir}",
        bar,
    ]
    print("\n".join(lines), flush=True)


# COCO 80 / VOC 20 class 이름 — TB 시각화 시 박스 label 표시용
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]
VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor",
]


def _draw_detections(img_tensor: torch.Tensor, pred_boxes_xyxy: torch.Tensor,
                     pred_scores: torch.Tensor, pred_classes: torch.Tensor,
                     gt_boxes_xyxy: torch.Tensor, gt_labels: torch.Tensor,
                     class_names: list[str], mean: list[float], std: list[float],
                     score_thresh: float = 0.3) -> torch.Tensor:
    """un-normalize → PIL draw (GT=green, pred=red) → tensor [3,H,W] uint8 → TB.

    args:
      img_tensor: [3,H,W] normalized (mean/std).
      pred_boxes_xyxy: [N,4] in current image px.
      pred_scores: [N], pred_classes: [N].
      gt_boxes_xyxy: [M,4] in current image px (transform 후 — orig 아닌 *transformed* 좌표).
    """
    m = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
    s = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)
    img_unnorm = (img_tensor.detach().cpu() * s.cpu() + m.cpu()).clamp(0, 1)
    pil = Image.fromarray((img_unnorm.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    draw = ImageDraw.Draw(pil)
    # GT (green)
    for j in range(gt_boxes_xyxy.shape[0]):
        x1, y1, x2, y2 = gt_boxes_xyxy[j].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=2)
        ln = int(gt_labels[j])
        name = class_names[ln] if ln < len(class_names) else str(ln)
        draw.text((x1, max(y1 - 10, 0)), f"GT:{name}", fill="lime")
    # pred (red) — score>=thresh 만
    for j in range(pred_boxes_xyxy.shape[0]):
        sc = float(pred_scores[j])
        if sc < score_thresh:
            continue
        x1, y1, x2, y2 = pred_boxes_xyxy[j].tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        cn = int(pred_classes[j])
        name = class_names[cn] if cn < len(class_names) else str(cn)
        draw.text((x1, min(y2 + 2, pil.size[1] - 10)), f"{name}:{sc:.2f}", fill="red")
    return to_tensor(pil)  # [3,H,W] float [0,1]


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    if cfg.seed is None or str(cfg.seed) == "???":
        raise RuntimeError("seed required (CLAUDE.md CRITICAL). e.g. seed=42")

    seed = int(cfg.seed)
    _set_seed(seed)

    out_dir = Path(cfg.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config.yaml")
    (out_dir / "seed.txt").write_text(f"{seed}\n")
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        rev = "unknown"
    (out_dir / "git_rev.txt").write_text(rev + "\n")

    device = torch.device(cfg.device)
    train_loader, eval_loader = _build_loaders(cfg, seed)

    # model.num_classes 가 ${data.num_classes} interpolation 일 수 있어 force-set
    cfg.model.num_classes = cfg.data.num_classes
    model = build_diffusiondet(cfg.model).to(device)
    criterion = build_criterion(cfg.data.num_classes, cfg.loss).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim_cfg = cfg.train.optimizer
    optim = torch.optim.AdamW(trainable, lr=float(optim_cfg.lr),
                              weight_decay=float(optim_cfg.weight_decay))

    sch_cfg = cfg.train.scheduler
    multistep = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=list(sch_cfg.milestones), gamma=float(sch_cfg.gamma),
    )
    warmup_iters = int(sch_cfg.get("warmup_iters", 0))
    warmup_factor = float(sch_cfg.get("warmup_factor", 1.0))

    # AMP dtype 분기 — cfg.train.amp 와 cfg.train.amp_dtype 의 곱집합.
    # - bf16 (Blackwell native): dynamic range 가 fp32 와 동일 → GradScaler 불필요. fp16 overflow 의 NaN 회피.
    # - fp16: GradScaler 필수 (loss scaling 으로 작은 grad 보존 + inf 발생 시 step skip).
    # - fp32 (amp=false): autocast 자체 비활성, scaler 도 no-op.
    # 이전 fp16 학습이 iter 2825/5825 에서 NaN 으로 죽은 후 bf16 으로 전환.
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    amp_dtype_str = str(cfg.train.get("amp_dtype", "float16")).lower()
    if amp_dtype_str in ("bfloat16", "bf16"):
        amp_dtype = torch.bfloat16
        use_scaler = False
    elif amp_dtype_str in ("float16", "fp16", "half"):
        amp_dtype = torch.float16
        use_scaler = amp_enabled
    else:
        amp_dtype = torch.float32
        use_scaler = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # ─── resume (선택) ───────────────────────────────────────────────
    # cfg.train.resume = <ckpt path> 있으면 model/optim/sched/scaler 복원 + start_epoch 갱신.
    # metrics.csv / eval_history.json 은 새 run_dir 의 새 파일로 시작 (이전 run_dir 은 그대로 보존).
    resume_path = cfg.train.get("resume", None)
    start_epoch = 0
    iter_count = 0
    if resume_path:
        rp = Path(resume_path)
        if not rp.exists():
            raise FileNotFoundError(f"resume ckpt not found: {rp}")
        state = torch.load(rp, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        multistep.load_state_dict(state["scheduler"])
        if "scaler" in state and use_scaler:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state.get("epoch", -1)) + 1
        iter_count = int(state.get("iter", 0))
        print(f"[resume] {rp} → start_epoch={start_epoch} iter_count={iter_count}")

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "iter", "loss_total", "loss_cls", "loss_l1", "loss_giou",
            "grad_norm", "lr", "elapsed_sec",
        ])
    tb_writer = SummaryWriter(log_dir=str(out_dir / "logs"))

    grad_clip = float(cfg.train.grad_clip)
    log_interval = int(cfg.train.log_interval)
    epochs = int(cfg.train.epochs)
    max_iters = int(cfg.train.get("max_iters", 0))  # 0 = unlimited (정식 학습)

    t_start = time.monotonic()
    last_loss = None
    finished = False
    best_map = -1.0
    eval_history: list[dict] = []
    eval_interval = int(cfg.train.get("eval_interval_epoch", 1))
    n_vis = int(cfg.train.get("eval_vis_samples", 4))  # epoch eval 시 TB 에 박을 이미지 수
    class_names = COCO_CLASSES if cfg.data.name == "coco" else VOC_CLASSES

    # ─── startup log — 학습 핵심 설정 한 화면 요약 (재현 용이) ───
    _print_startup(cfg, out_dir, rev, len(train_loader.dataset), len(eval_loader.dataset),
                   device, amp_enabled, amp_dtype_str, warmup_iters, warmup_factor, eval_interval)

    def _run_eval(epoch_id: int) -> dict:
        model.eval()
        if cfg.data.name == "coco":
            m = coco_eval(model, eval_loader, device, verbose=False)
        else:
            m = voc_eval(model, eval_loader, device, num_classes=cfg.data.num_classes)
        model.train()
        return m

    @torch.no_grad()
    def _log_detection_images(epoch_id: int, n: int = 4):
        """eval_loader 의 첫 n 개 이미지에 모델 inference 후 TB add_image.
        GT (lime) + pred (red, score>=0.3) overlay. transformed-image 좌표 그대로 그림.
        """
        model.eval()
        added = 0
        for batch in eval_loader:
            images, targets = batch
            images = images.to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                out = model(images)
            scores = out["pred_logits"].sigmoid()
            boxes = out["pred_boxes"]
            B, N, C = scores.shape
            for b in range(B):
                if added >= n:
                    break
                tgt = targets[b]
                # transformed-image 좌표는 image_tensor 의 H,W 와 일치 (no rescale)
                cur_h, cur_w = tgt["size"]
                # GT 도 transformed 좌표 (transforms 가 이미 resize 적용)
                gt_boxes = tgt["boxes"].cpu()
                gt_labels = tgt["labels"].cpu()
                # top-K pred per image
                scs = scores[b].reshape(-1)
                top_scores, top_idx = scs.topk(min(20, scs.numel()))
                box_idx = top_idx // C
                cls_idx = top_idx % C
                pred_b = boxes[b][box_idx].cpu()
                img_grid = _draw_detections(
                    images[b].cpu(), pred_b, top_scores.cpu(), cls_idx.cpu(),
                    gt_boxes, gt_labels, class_names,
                    list(cfg.data.mean), list(cfg.data.std), score_thresh=0.3,
                )
                tb_writer.add_image(f"eval/sample_{added}", img_grid, epoch_id)
                added += 1
            if added >= n:
                break
        model.train()

    for epoch in range(start_epoch, epochs):
        if finished:
            break
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            images, targets = batch
            images = images.to(device, non_blocking=True)
            targets = _targets_to_device(targets, device)

            # === Mixed Precision (autocast) — 핵심 설명 ==========================
            # `torch.amp.autocast` 는 *그 안에서 실행되는 op 만* 자동으로 dtype 캐스트.
            # - model param 자체는 fp32 그대로 유지 (메모리에 fp32 한 벌만 있음).
            # - 안에서 호출된 conv/linear/matmul 같은 "안전한 무거운 op" → bf16/fp16 으로
            #   캐스트하여 실행 (Tensor Core 활용 → 1.5-2x 가속, activation 메모리 절반).
            # - 반대로 reduction (mean/var/softmax sum, loss 합산 등) 처럼 numerical
            #   stability 가 민감한 op 는 fp32 로 *유지* (autocast 의 op-별 화이트리스트).
            # - 결과로 loss tensor 의 dtype 은 보통 fp32 (마지막 reduction 후).
            # - backward 는 autocast context *밖* 에서 호출 — 그래야 grad 가 fp32 로 누적.
            #   computation graph 가 dtype 따라 grad dtype 도 정해짐.
            # - dtype=bfloat16: range 가 fp32 와 같음(exp 8 bit) → overflow/underflow X.
            #   dtype=float16:    range 좁음(exp 5 bit) → 작은 grad underflow → GradScaler 로 보완.
            # - enabled=False: context 가 no-op 와 같음 (모두 fp32 로 실행).
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model(images, targets)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["loss_total"]
            # NaN/Inf loss skip — fp16 overflow / 데이터 edge case 등.
            # 본 DiffusionDet repo 도 동일 패턴 (assertion 대신 skip).
            if not torch.isfinite(loss).all():
                print(f"WARN: non-finite loss at epoch {epoch} iter {iter_count} (skipping)")
                optim.zero_grad(set_to_none=True)
                continue

            # LR warmup — cold-start 의 grad explosion 완화. iter 0 → warmup_iters 동안
            # lr 가 (warmup_factor × base_lr) → base_lr 로 linear 증가.
            if warmup_iters > 0 and iter_count < warmup_iters:
                alpha = iter_count / max(warmup_iters, 1)
                w = warmup_factor + (1.0 - warmup_factor) * alpha
                for pg in optim.param_groups:
                    pg["lr"] = float(optim_cfg.lr) * w

            # === backward / step — AMP dtype 별 분기 ============================
            # use_scaler=True (fp16 만): loss 에 큰 상수 scale 곱해 backward → 작은 grad 가
            # fp16 underflow 안 되도록. 이후 unscale_ 로 원래 크기 복원 + clip + step.
            # scaler.step() 은 unscale 후 inf/NaN grad 감지하면 *자동 skip* (param 보호).
            # use_scaler=False (bf16/fp32): 그냥 backward → clip → step. 보조장치 불필요.
            optim.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=grad_clip)
                optim.step()

            iter_count += 1
            last_loss = float(loss.detach())
            cur_lr = optim.param_groups[0]["lr"]
            cls_v = float(loss_dict["loss_cls"])
            l1_v = float(loss_dict["loss_l1"])
            giou_v = float(loss_dict["loss_giou"])
            gnorm = float(grad_norm)
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, iter_count, last_loss, cls_v, l1_v, giou_v,
                    gnorm, cur_lr, round(time.monotonic() - t_start, 2),
                ])
            tb_writer.add_scalar("train/loss_total", last_loss, iter_count)
            tb_writer.add_scalar("train/loss_cls", cls_v, iter_count)
            tb_writer.add_scalar("train/loss_l1", l1_v, iter_count)
            tb_writer.add_scalar("train/loss_giou", giou_v, iter_count)
            # AMP 첫 step grad_norm NaN 은 skip (TB 가 NaN 처리 못 함)
            if gnorm == gnorm:
                tb_writer.add_scalar("train/grad_norm", gnorm, iter_count)
            tb_writer.add_scalar("train/lr", cur_lr, iter_count)
            if iter_count % log_interval == 0 or iter_count == 1:
                # 메모리 모니터링 — OOM kill 직전의 마지막 print 가 peak 흔적이 됨.
                # rss = 컨테이너 안 python process 가 쓰는 host RAM (anonymous + shared).
                # host_avail = 호스트 RAM 의 available (kernel 의 free + reclaimable buffer/cache).
                # gpu_alloc / gpu_peak = torch CUDA allocator 의 현재 / 역대 peak (MB).
                rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1e9
                vm = psutil.virtual_memory()
                host_used_gb = vm.used / 1e9
                host_avail_gb = vm.available / 1e9
                host_swap_used_gb = psutil.swap_memory().used / 1e9
                gpu_alloc_gb = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                gpu_peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                print(f"[epoch {epoch} iter {iter_count}] loss={last_loss:.4f} "
                      f"grad_norm={gnorm:.2f} lr={cur_lr:.2e}  "
                      f"| rss={rss_gb:.1f}G host_avail={host_avail_gb:.1f}G "
                      f"swap={host_swap_used_gb:.1f}G gpu={gpu_alloc_gb:.1f}/{gpu_peak_gb:.1f}G", flush=True)
                tb_writer.add_scalar("system/rss_gb", rss_gb, iter_count)
                tb_writer.add_scalar("system/host_avail_gb", host_avail_gb, iter_count)
                tb_writer.add_scalar("system/host_swap_gb", host_swap_used_gb, iter_count)
                tb_writer.add_scalar("system/gpu_alloc_gb", gpu_alloc_gb, iter_count)
                tb_writer.add_scalar("system/gpu_peak_gb", gpu_peak_gb, iter_count)
            # epoch 내에서도 1000 iter 마다 last.pt 저장 (crash 시 손실 최소화)
            if iter_count % 1000 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": multistep.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "iter": iter_count,
                }, ckpt_dir / "last.pt")
            if max_iters > 0 and iter_count >= max_iters:
                finished = True
                break
        multistep.step()
        # ckpt — last.pt 매 epoch
        ckpt_state = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": multistep.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "iter": iter_count,
        }
        torch.save(ckpt_state, ckpt_dir / "last.pt")

        # epoch eval
        if eval_interval > 0 and (epoch + 1) % eval_interval == 0 and not finished:
            print(f"[epoch {epoch}] running eval ...")
            t_eval = time.monotonic()
            try:
                m = _run_eval(epoch)
            except Exception as e:
                print(f"WARN: eval failed at epoch {epoch}: {e}")
                m = {"metric_primary": -1.0, "error": str(e)}
            eval_sec = round(time.monotonic() - t_eval, 1)
            mp = float(m.get("metric_primary", -1.0))
            tb_writer.add_scalar("eval/metric_primary", mp, epoch)
            for k in ("AP50", "AP75", "APs", "APm", "APl", "mAP50"):
                if k in m:
                    tb_writer.add_scalar(f"eval/{k}", float(m[k]), epoch)
            print(f"[epoch {epoch}] eval metric_primary={mp:.4f} ({eval_sec}s)")
            eval_history.append({"epoch": epoch, "iter": iter_count,
                                  "eval_sec": eval_sec, **m})
            (out_dir / "eval_history.json").write_text(
                json.dumps(eval_history, indent=2))
            if mp > best_map:
                best_map = mp
                torch.save(ckpt_state, ckpt_dir / "best.pt")
                print(f"[epoch {epoch}] best.pt saved (metric_primary={mp:.4f})")
            # eval 후 sample 이미지 TB 에 박기 (학습 진행 가시화)
            try:
                _log_detection_images(epoch, n=n_vis)
            except Exception as e:
                print(f"WARN: log_detection_images failed: {e}")

    tb_writer.close()
    print(f"Done. epoch={epoch} iter={iter_count} last_loss={last_loss} runs/{out_dir.name}")


if __name__ == "__main__":
    main()
