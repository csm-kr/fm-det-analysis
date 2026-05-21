import os
import json
import torch 
import tempfile
import numpy as np
from pycocotools.cocoeval import COCOeval

class Evaluator(object):
    def __init__(self, data_type='coco', coco_ids=None):
        self.data_type = data_type
        self.coco_ids = coco_ids

        # COCO 평가를 위한 결과 저장소
        self.results = []
        self.img_ids = []

    def get_info(self, info):
        if self.data_type == 'coco':
            (pred_boxes, pred_labels, pred_scores, img_id, img_info) = info
            
            # 1. img_id가 Tensor일 경우를 대비해 int로 변환 (JSON 호환)
            if isinstance(img_id, torch.Tensor):
                img_id = int(img_id.item())
            self.img_ids.append(img_id)

            # 2. 이미지 크기 정보
            w_img = img_info['width']
            h_img = img_info['height']

            # 3. GPU Tensor -> CPU Numpy로 일괄 변환 (속도 향상 핵심)
            # clone()을 해줘야 원본 텐서에 영향을 주지 않음
            if pred_boxes.device.type != 'cpu':
                boxes_np = pred_boxes.detach().cpu().numpy()
                labels_np = pred_labels.detach().cpu().numpy()
                scores_np = pred_scores.detach().cpu().numpy()
            else:
                boxes_np = pred_boxes.detach().numpy()
                labels_np = pred_labels.detach().numpy()
                scores_np = pred_scores.detach().numpy()

            # 4. 좌표 변환 (XYXY Normalized -> XYWH Absolute)
            # boxes_np : [x1, y1, x2, y2] (0~1 scale)
            
            # Width & Height 계산 (Normalized)
            boxes_np[:, 2] -= boxes_np[:, 0] # w = x2 - x1
            boxes_np[:, 3] -= boxes_np[:, 1] # h = y2 - y1
            
            # Scale 복원 (Absolute)
            boxes_np[:, 0] *= w_img # x1
            boxes_np[:, 1] *= h_img # y1
            boxes_np[:, 2] *= w_img # w
            boxes_np[:, 3] *= h_img # h

            # 5. 결과 리스트에 추가
            # Numpy 배열을 순회하는 것이 GPU Tensor 순회보다 훨씬 빠름
            for box, label, score in zip(boxes_np, labels_np, scores_np):
                label_idx = int(label)
                
                # Background 클래스 제외 (보통 80번이 배경이거나 padding인 경우)
                if label_idx == 80: 
                    continue

                coco_result = {
                    'image_id': img_id,
                    'category_id': self.coco_ids[label_idx] if self.coco_ids else label_idx,
                    'score': float(score),
                    'bbox': box.tolist(), # [x, y, w, h]
                }
                self.results.append(coco_result)

    def evaluate(self, dataset):
        if self.data_type == 'coco':
            # 결과가 하나도 없을 경우 예외 처리
            if len(self.results) == 0:
                print("No detections found! mAP is 0.")
                return 0.0

            # 1. 임시 JSON 파일 생성
            fd, tmp = tempfile.mkstemp()
            os.close(fd) # 파일 디스크립터 닫기 (안전)

            try:
                # JSON 저장
                with open(tmp, 'w') as f:
                    json.dump(self.results, f)

                # 2. COCO API 로드
                cocoGt = dataset.coco
                cocoDt = cocoGt.loadRes(tmp) # 예측 결과 로드

                # 3. 평가 수행
                coco_eval = COCOeval(cocoGt=cocoGt, cocoDt=cocoDt, iouType='bbox')
                coco_eval.params.imgIds = self.img_ids
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                mAP = coco_eval.stats[0] # mAP @ IoU=0.5:0.95
                
                return mAP
            
            except Exception as e:
                print(f"Evaluation Error: {e}")
                return 0.0
            
            finally:
                # 4. 임시 파일 삭제 (중요: 디스크 용량 보호)
                if os.path.exists(tmp):
                    os.remove(tmp)
        
        return 0.0


def update_evaluator(results, batch, evaluator):
    for j, output in enumerate(results):
        
        # --- A. 메타데이터 추출 ---
        # Detectron2 스타일의 batch는 List[Dict] 형태라고 가정
        img_id = batch[j]['image_id']
        orig_h, orig_w = batch[j]['orig_size']
        
        img_info = {'width': orig_w, 'height': orig_h}

        # --- B. 예측 결과 처리 ---
        # output.image_size : 모델 입력으로 들어간 리사이즈된 크기 (H, W)
        h, w = output.image_size 
        
        # 정규화를 위한 divisor (w, h, w, h)
        image_size_whwh = torch.tensor([w, h, w, h], dtype=torch.float32, device=output.pred_boxes.tensor.device)

        # Box Normalization (0~1 사이 값으로 변환)
        # clone()을 사용하지 않으면 output 내부 값이 바뀔 수 있으므로 주의 (여기선 나눗셈이라 새 텐서 생성됨)
        pred_boxes = output.pred_boxes.tensor / image_size_whwh
        pred_scores = output.scores 
        pred_labels = output.pred_classes 

        # --- C. Evaluator 업데이트 ---
        info = (pred_boxes, pred_labels, pred_scores, img_id, img_info)
        evaluator.get_info(info)
    
    return evaluator