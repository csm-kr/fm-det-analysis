import sys, os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR
from tqdm import tqdm

# 프로젝트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.dataset.coco_dataset import COCO_Dataset, build_train_augmentation, build_test_augmentation
from src.model.diffusiondet import DiffusionDet
from src.loss.diffusion_loss import DiffusionDetLoss
from src.evaluation.evaluator import Evaluator, update_evaluator
from detectron2.structures import ImageList

# --- Helper Functions ---

def build_lr_scheduler(optimizer, iters_per_epoch, warmup_iters=1000, milestones_epochs=[47, 57], gamma=0.1):
    milestones_iters = [m * iters_per_epoch for m in milestones_epochs]
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters)
    main_milestones = [m - warmup_iters for m in milestones_iters]
    main_scheduler = MultiStepLR(optimizer, milestones=main_milestones, gamma=gamma)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])
    return scheduler

class DataPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
        pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        self.normalizer = lambda x: (x - self.pixel_mean) / self.pixel_std

    def preprocess_data(self, targets, device):
        ret_images = []
        ret_targets = []
        for target in targets:
            image = target["image"].to(device)
            ret_images.append(self.normalizer(image))
            gt_instances = target["instances"].to(device)
            new_target = {
                "boxes": gt_instances.gt_boxes.tensor,
                "labels": gt_instances.gt_classes,
                "image_size_whwh": torch.as_tensor([gt_instances.image_size[1], gt_instances.image_size[0], 
                                                    gt_instances.image_size[1], gt_instances.image_size[0]], 
                                                   dtype=torch.float, device=device)
            }
            ret_targets.append(new_target)
        ret_images = ImageList.from_tensors(ret_images, 32)
        return ret_images, ret_targets

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def init_for_distributed(rank, opts):
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{opts.port}',
                            world_size=opts.world_size, rank=opts.rank)
    torch.distributed.barrier()
    setup_for_distributed(opts.rank == 0)
    return local_gpu_id

# --- Main Worker ---

def main_worker(rank, opts):

    # init
    device = init_for_distributed(rank, opts)
    ckpt_dir = opts.ckpt_dir
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        writer = SummaryWriter("./runs")
    else:
        writer = None

    # 2. Dataset and Dataloader
    train_dataset = COCO_Dataset(data_root=opts.data_root, split="train", transform=build_train_augmentation())
    test_dataset = COCO_Dataset(data_root=opts.data_root, split="val", transform=build_test_augmentation())

    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opts.batch_size // opts.world_size,
        shuffle=False,
        num_workers=opts.num_workers // opts.world_size,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn, # [수정] 에러 방지 필수
        pin_memory=True
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=test_dataset.collate_fn)

    # 3. Model,
    model = DiffusionDet(scale=opts.signal_scale, box_renewal=True).to(device)
    model = DDP(module=model, device_ids=[device], broadcast_buffers=False)
    data_preprocessor = DataPreprocessor().to(device)

    # 4. Criterion
    criterion = DiffusionDetLoss()

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)

    # 6. Scheduler
    scheduler = build_lr_scheduler(optimizer, iters_per_epoch=len(train_loader))

    start_epoch = 0
    global_iter = 0
    best_map = 0.0

    if opts.resume:
        # 불러올 체크포인트 경로를 직접 지정 (기본적으로 last.pth 사용)
        resume_path = os.path.join(ckpt_dir, "last.pth")
        
        if os.path.isfile(resume_path):
            print(f"=> Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=f'cuda:{device}')
            
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint['best_mAP']
            
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            global_iter = start_epoch * len(train_loader)
            
            print(f"=> Loaded checkpoint (epoch {checkpoint['epoch']})")
        else:
            print(f"=> No checkpoint found at {resume_path}, starting from scratch.")

    # 7. Train
    print("Start Training...")
    for epoch in range(start_epoch, opts.num_epochs):
        train_sampler.set_epoch(epoch) 
        model.train()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=(rank != 0))
        for i, batch in pbar:
            images, targets = data_preprocessor.preprocess_data(batch, device)
            pred_logits, pred_boxes = model(images, targets)
            loss_dict = criterion(pred_logits, pred_boxes, targets)
            losses = sum(loss_dict[k] for k in loss_dict.keys())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            global_iter += 1
            if rank == 0 and i % opts.vis_step == 0:
                writer.add_scalar('Total Loss', losses.item(), global_step=global_iter)
                pbar.set_description(f"Epoch [{epoch+1}/{opts.num_epochs}] Loss: {losses.item():.4f}")

        # 8. Eval 
        dist.barrier() # 모든 프로세스 학습 완료 대기
        if rank == 0:
            model.eval()
            evaluator = Evaluator(data_type='coco', coco_ids=test_dataset.coco_ids)
            for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc="Eval"):
                images, targets = data_preprocessor.preprocess_data(batch, device)
                with torch.no_grad():
                    results = model(images, targets)
                evaluator = update_evaluator(results, batch, evaluator)

            current_mAP = evaluator.evaluate(test_loader.dataset)
            writer.add_scalar('MAP', current_mAP, global_step=epoch)

            is_best = current_mAP > best_map
            if is_best:
                best_map = current_mAP

            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mAP': best_map,
            }

            torch.save(checkpoint_data, os.path.join(ckpt_dir, "last.pth"))

            if is_best:
                torch.save(checkpoint_data, os.path.join(ckpt_dir, "best.pth"))
                print(f"🔥 New Best: {best_map:.4f}")

    if rank == 0: 
        writer.close()
    dist.destroy_process_group()

# --- Main Entry ---

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=2.5e-5)
    parser.add_argument('--data_root', type=str, default=r'/usr/src/data/coco')
    parser.add_argument('--num_epochs', type=int, default=61)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--vis_step', type=int, default=20)
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    parser.add_argument('--port', type=int, default=23456)
    parser.add_argument('--signal_scale', type=float, default=2.0)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', action='store_true', help='resume from the last checkpoint in ckpt_dir')
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DiffusionDet training', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    # 1. World Size 설정 (중복 파싱 제거)
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = opts.world_size * 4

    # 2. Spawn
    mp.spawn(main_worker, args=(opts,), nprocs=opts.world_size, join=True)