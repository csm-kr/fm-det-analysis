import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import cv2
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from detectron2.structures import ImageList
from detectron2.layers import batched_nms

# 프로젝트 모듈 임포트
from src.dataset.coco_dataset import COCO_Dataset
from src.model.diffusiondet import DiffusionDet

from script.train import DataPreprocessor

from src.utils.label_info import coco_color_array, coco_label_list

from tqdm import tqdm

from src.evaluation.evaluator import Evaluator, update_evaluator
from src.dataset.coco_dataset import build_test_augmentation


# ====================================================
# 1. Visualization Utility (시각화 함수)
# ====================================================
def visualize_prediction(image_tensors, results, save_dir, img_idx, threshold=0.35):
    """
    image_tensors: (B, C, H, W) 또는 List[Tensor]
    results: [Instances, ...] 형태의 리스트
    save_dir: 이미지가 저장될 디렉토리 경로
    """
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)

    # 1. 역정규화를 위한 설정
    # (이미지별로 device가 다를 수 있으므로 루프 안에서 처리하거나 첫 번째 이미지 기준 설정)
    pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).view(3, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.120, 57.375]).view(3, 1, 1)

    # 2. Results 리스트를 순회
    for i, output in enumerate(results):
        # 개별 이미지 텐서 가져오기
        image_tensor = image_tensors[i]
        device = image_tensor.device

        # 이미지 역정규화 및 numpy 변환
        img = image_tensor.cpu() * pixel_std + pixel_mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 3. 해당 이미지의 예측 결과 추출
        boxes = output.pred_boxes.tensor.cpu()
        scores = output.scores.cpu()
        labels = output.pred_classes.cpu()

        # 4. 객체별 시각화
        for box, score, label in zip(boxes, scores, labels):
            if score < threshold:
                continue
                
            x1, y1, x2, y2 = map(int, box.tolist())
            label_idx = int(label)
            color = tuple(map(int, coco_color_array[label_idx]))
            caption = f"{coco_label_list[label_idx]} {score:.2f}"
            
            # 2. 메인 객체 바운딩 박스 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)

            # 3. 텍스트 사이즈 계산 (배경 박스 크기 결정용)
            # fontScale과 thickness는 취향에 따라 조절하세요.
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            font_thickness = 1
            text_size, baseline = cv2.getTextSize(caption, font_face, font_scale, font_thickness)
            
            # 텍스트 배경 박스 좌표 계산 (텍스트가 이미지 밖으로 나가지 않도록 y1 기준 처리)
            text_w, text_h = text_size
            back_res_rect = (x1, y1 - text_h - 10, x1 + text_w + 3, y1)
            
            # 4. 텍스트 배경 박스 그리기 (thickness=-1 이면 색상을 채웁니다)
            cv2.rectangle(img, 
                          (back_res_rect[0], back_res_rect[1]), 
                          (back_res_rect[2], back_res_rect[3]), 
                          color, 
                          -1)

            # 5. 텍스트 쓰기 (배경이 유색이므로 글자는 검정색(0,0,0) 혹은 흰색 추천)
            cv2.putText(img, 
                        caption, 
                        (x1 + 2, y1 - 5), 
                        font_face, 
                        font_scale, 
                        (0, 0, 0), # 글자 색상 (검정)
                        font_thickness, 
                        lineType=cv2.LINE_AA)

        # 5. 각 인덱스별로 파일명 생성하여 저장
        save_path = os.path.join(save_dir, f"result_{img_idx}.jpg")
        cv2.imwrite(save_path, img)
        print(f"Saved: {save_path}")


# ====================================================
# 4. Main Function
# ====================================================
def main():
    # 1. Config
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 1 # 테스트는 보통 1장씩 보거나 소량 배치
    checkpoint_path = "./checkpoints/last.pth" # 불러올 가중치 경로
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading Checkpoint: {checkpoint_path}")

    # 2. Dataset (Test/Val)
    dataset = COCO_Dataset(
        data_root=r"/usr/src/data/coco",
        split="val", # 또는 "test"
        transform=build_test_augmentation(), # 테스트용 트랜스폼 적용 필요
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # 순서대로 추론
        num_workers=4,
        collate_fn=dataset.collate_fn
    )

    data_preprocessor = DataPreprocessor().to(device)

    # 3. Model Build & Load
    model = DiffusionDet(scale=2.0, box_renewal=True).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # state_dict 키 불일치 방지 (DataParallel 등으로 저장된 경우)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        print("Model weights loaded successfully.")
    else:
        print("Checkpoint not found!")
        return

    # 4. Inference Loop
    print("Start Inference...")
    model.eval()

    # evaluation
    coco_ids = dataloader.dataset.coco_ids
    evaluator = Evaluator(data_type='coco', coco_ids=coco_ids) # 입력이 xyxy_norm 이 필요 

    for i, batch in tqdm(enumerate(dataloader)):

        # Preprocess
        images, targets = data_preprocessor.preprocess_data(batch, device)

        # train
        # pred_logits, pred_boxes = model(images, targets)

        # test
        # results = model(images, targets)
        with torch.no_grad(): # 검증 시 메모리 절약
            results = model(images, targets)
            # for vis
            # visualize_prediction(images, results, save_dir='./results', img_idx=i)
        evaluator = update_evaluator(results, batch, evaluator)

    mAP = evaluator.evaluate(dataloader.dataset)
    print("mAP : ", mAP)


if __name__ == "__main__":
    main()



    