import torch
import torch.nn as nn
from detectron2.config import get_cfg
from detectron2.structures import ImageList
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer 

class ResNet50FPN(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()

        # 1. Config 생성
        cfg = get_cfg()

        # -------------------------------------------------------------
        cfg.MODEL.DEVICE = device
        cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"  # FPN 백본 지정
        cfg.MODEL.RESNETS.DEPTH = 50
        cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False  # TorchVision 가중치는 이게 False여야 함
        cfg.MODEL.RESNETS.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
        # -------------------------------------------------------------

        # 2. Backbone 빌드
        self.backbone = build_backbone(cfg).to(device)
        # self.size_divisibility = 32

        # 3. 가중치 로드
        checkpointer = DetectionCheckpointer(self.backbone)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        
        self.device = device

    def forward(self, x):
        
        # images = ImageList.from_tensors(list(x), self.size_divisibility)
        outputs = self.backbone(x)

        # return outputs
        
        p2 = outputs["p2"]
        p3 = outputs["p3"]
        p4 = outputs["p4"]
        p5 = outputs["p5"]

        return [p2, p3, p4, p5]

# -------------------------------------------------------------------------
# 🧪 실험 및 테스트 코드
# -------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. 장치 설정 (GPU가 있으면 cuda, 없으면 cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📌 Current Device: {device}")

    # 2. 모델 초기화
    model = ResNet50FPN(device=device)
    model.eval() 

    # 3. 랜덤 이미지 생성 (Batch Size=2, Channel=3, Height=640, Width=640)
    image = torch.randn(2, 3, 800, 800).to(device)
    
    print(model(image)[0].shape)
    print(model(image)[1].shape)
    print(model(image)[2].shape)
    print(model(image)[3].shape)
