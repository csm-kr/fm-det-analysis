
import sys, os
# path 맞추기 위해서 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from detectron2.structures import Boxes, Instances

from src.utils.bbox_utils import box_xyxy_to_cxcywh
from src.utils.vis_utils import visualize_instances_d2

from detectron2.data import transforms as T

class COCO_Dataset(Dataset):
    def __init__(
        self,
        data_root="/usr/src/data/coco",
        split="train",
        transform=None,
        visualization=False,
    ):
        """
        Detectron2 스타일의 객체 탐지를 위한 커스텀 COCO 데이터셋 로더.

        사용하는 Detectron2 Class : 
            detectron2.structures.Boxes, detectron2.structures.Instances

            참고 instances = Instances(image_size)  
                instances.gt_boxes = boxes        # 정답 바운딩 박스 (detectron2.structures.Boxes, xyxy)
                instances.gt_classes = labels     # 정답 클래스 ID (Tensor[int])

        사용하는 Data Augmentation : 
            Detectron2 스타일의 데이터 증강 파이프라인.
            detectron2.data.transforms.AugmentationList
            e.g) from detectron2.data import transforms as T

        시각화 (visualization) (bool): 
            True일 경우, 변환(augmentation) 이후의 이미지에 
            정답 바운딩 박스와 라벨을 시각화하여 디버깅/확인용으로 표시함.

        데이터 폴더 구조 (data structure):
            data_root/
                ├── train2017/*.jpg
                ├── val2017/*.jpg
                ├── annotations/
                │     ├── instances_train2017.json
                │     └── instances_val2017.json

        매개변수 (Args):
            data_root (str): COCO 데이터셋의 루트 경로.
            split (str): 사용할 데이터셋 분할 이름. ['train', 'val'] 중 하나.
            transform (detectron2.data.transforms.AugmentationList, optional):
                
                예시: RandomFlip, ResizeShortestEdge, RandomCrop 등.

        리턴 (returns) : dictionary
            {
                "file_name": str,                    # 이미지 경로
                "height": int,                       # 이미지 높이
                "width": int,                        # 이미지 너비
                "image_id": int,                     # 이미지 ID
                "image": Tensor,                     # 이미지 데이터 (torch.Tensor)
                "instances": detectron2.structures.Instances  # GT 박스 및 클래스 정보
                "orig_size": (h, w)
            }

        instances.gt_boxes = boxes        # 정답 바운딩 박스 (detectron2.structures.Boxes, xyxy)
        instances.gt_classes = labels     # 정답 클래스 ID (Tensor[int])

        """
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.data_root = data_root
        self.split = split
        self.set_name = split + "2017"
        self.transform = transform

        self.visualization = visualization

        # if download:
        #     download_coco(root_dir=data_root)

        ann_path = os.path.join(
            self.data_root, "annotations", f"instances_{self.set_name}.json"
        )
        self.coco = COCO(ann_path)
        self.ids = self.coco.getImgIds()
        self.catid_to_contig = {cat_id: i for i, cat_id in enumerate(self.coco.getCatIds())}
        self.coco_ids = sorted(self.coco.getCatIds()) 

        print(f"[COCO_Dataset] Loaded {len(self.ids)} images from {self.set_name}")

    # -------------------------------------------------------------
    # Load functions
    # -------------------------------------------------------------
    def _load_image(self, id):
        img_info = self.coco.loadImgs(id)[0]
        path = img_info["file_name"]
        img_path = os.path.join(self.data_root, self.set_name, path)
        image = Image.open(img_path).convert("RGB")
        return image

    def _load_anno(self, id):
        ann_ids = self.coco.getAnnIds(imgIds=id)
        return self.coco.loadAnns(ann_ids)

    # -------------------------------------------------------------
    # Core
    # -------------------------------------------------------------
    def __getitem__(self, index):
        image_id = self.ids[index]
        img_info = self.coco.loadImgs(image_id)[0]

        # ---------------------------------------------------------
        # Load image
        # ---------------------------------------------------------
        file_name = os.path.join(self.data_root, self.set_name, img_info["file_name"])
        image = self._load_image(image_id)  # PIL.Image
        width, height = image.size
        orig_size = (height, width)

        # ---------------------------------------------------------
        # Parse annotation
        # ---------------------------------------------------------
        boxes, labels = self.parse_coco(self._load_anno(image_id), type="bbox")

        if boxes is None or len(boxes) == 0:
            # skip empty annotation
            return self.__getitem__((index + 1) % len(self))

        # ---------------------------------------------------------
        # Transform (Detectron2 style)
        # ---------------------------------------------------------
        if self.transform is not None:
            image_np = np.asarray(image)
            aug_input = T.AugInput(image_np, boxes=boxes.tensor)
            _ = self.transform(aug_input)

            boxes_tensor = torch.as_tensor(aug_input.boxes)

            # Filter invalid / small boxes
            wh = boxes_tensor[:, 2:] - boxes_tensor[:, :2]
            valid = (wh[:, 0] > 2) & (wh[:, 1] > 2)
            boxes_tensor = boxes_tensor[valid]
            labels = labels[valid]

            # skip if no valid boxes remain
            if boxes_tensor.shape[0] == 0:
                return self.__getitem__((index + 1) % len(self))

            # transformed image and boxes
            boxes = Boxes(boxes_tensor)
            image = aug_input.image  
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.uint8) # numpy [H, W, C] --> torch [C, H, W]
        else:
            # no transform → use original image
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)), dtype=torch.uint8) # numpy [H, W, C] --> torch [C, H, W]

        # get tramsformed size
        _, h, w = image.shape
        # ---------------------------------------------------------
        # Clamp boxes to image size
        # ---------------------------------------------------------
        boxes.tensor[:, 0::2].clamp_(min=0, max=w)
        boxes.tensor[:, 1::2].clamp_(min=0, max=h)

        # Remove degenerate boxes (x1>=x2 or y1>=y2)
        valid = (boxes.tensor[:, 2] > boxes.tensor[:, 0]) & (boxes.tensor[:, 3] > boxes.tensor[:, 1])
        boxes.tensor = boxes.tensor[valid]
        labels = labels[valid]

        if boxes.tensor.shape[0] == 0:
            return self.__getitem__((index + 1) % len(self))

        # ---------------------------------------------------------
        # Build Instances object
        # ---------------------------------------------------------
        image_size = (h, w)  # updated size after transform
        instances = Instances(image_size)
        instances.gt_boxes = boxes
        instances.gt_classes = labels

        # ---------------------------------------------------------
        # Build final dict (Detectron2-style)
        # ---------------------------------------------------------
        data_dict = {
            "file_name": file_name,
            "height": h,
            "width": w,
            "image_id": image_id,
            "image": image,         # tensor [3,H,W]
            "instances": instances, # Detectron2 Instances (Boxes, )
            "orig_size" : orig_size
        }

    # ---------------------------------------------------------
    # Visualization 
    # ---------------------------------------------------------
        if self.visualization:
            visualize_instances_d2(
                image,
                instances,
                data_type="coco",
                num_labels=91,
                name=f"sample_{image_id}.jpg",
                save=False
            )

        return data_dict
    
    # -------------------------------------------------------------
    # Annotation Parsing
    # -------------------------------------------------------------
    def parse_coco(self, anno, type="bbox"):
        """
        Parse COCO-style annotations and return valid bounding boxes & labels.
        Supports object detection format only.

        Returns:
            boxes (torch.FloatTensor): (N, 4) in [x1, y1, x2, y2]
            labels (torch.LongTensor): (N,) - continus integer
        """
        if type != "bbox":
            raise NotImplementedError("Only 'bbox' type supported for COCO parser.")

        boxes = []
        labels = []

        for a in anno:
            # skip annotations without box or marked as crowd
            if "bbox" not in a or a.get("iscrowd", 0) == 1:
                continue

            x, y, w, h = a["bbox"]
            # filter invalid or degenerate boxes
            if w < 1 or h < 1:
                continue

            # convert xywh → xyxy
            boxes.append([x, y, x + w, y + h])

            # convert category_id → contiguous index (detectron2 uses 0-based)
            cat_id = a["category_id"]
            labels.append(self.catid_to_contig.get(cat_id, -1))

        # handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # filter invalid coords (x2>x1, y2>y1)
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]

        # wrap as detectron2 Boxes for consistency
        boxes = Boxes(boxes)

        return boxes, labels

    def __len__(self):
        return len(self.ids)
    
    def collate_fn(self, batch):
        return batch   # list(dict) 그대로 반환


def build_train_augmentation():
    return T.AugmentationList([
        T.RandomFlip(horizontal=True, vertical=False),

        T.ResizeShortestEdge(short_edge_length=[400, 500, 600],
                             sample_style="choice", max_size=1333),

        T.RandomApply(
            T.RandomCrop(crop_type="absolute_range", crop_size=(384, 600)),
            prob=0.5
        ),

        T.ResizeShortestEdge(short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                             sample_style="choice", max_size=1333),

    ])

def build_test_augmentation():
    return T.AugmentationList([
        T.ResizeShortestEdge(short_edge_length=[800], sample_style="choice", max_size=1333),
    ])

# test code
if __name__ == "__main__":
    import random, numpy as np, torch

    # Apply resize → crop → final resize (Detectron2 standard)
    # ✅ 매 실행마다 다른 랜덤 시드 설정
    np.random.seed(None)
    seed = np.random.randint(0, 2**32 - 1)
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"🌱 Random seed set to: {seed}")

    def build_train_augmentation():
        return T.AugmentationList([
            T.RandomFlip(horizontal=True, vertical=False),

            T.ResizeShortestEdge(short_edge_length=[400, 500, 600],
                                 sample_style="choice", max_size=1333),

            T.RandomApply(
                T.RandomCrop(crop_type="absolute_range", crop_size=(384, 600)),
                prob=0.5
            ),

            T.ResizeShortestEdge(short_edge_length=(480, 512, 544, 576, 608, 640,
                                                    672, 704, 736, 768, 800), 
                                                    sample_style="choice", max_size=1333),

        ])

    def build_test_augmentation():
        return T.AugmentationList([
            T.ResizeShortestEdge(short_edge_length=[800], sample_style="choice", max_size=1333),
        ])
    
    transform_train = build_train_augmentation()
    transform_test = build_test_augmentation()

    dataset = COCO_Dataset(
        data_root="/usr/src/data/coco",
        split="val",
        transform=transform_train,
        visualization=True,
    )

    for i, data_dict in enumerate(dataset):

        image = data_dict["image"]               # torch.Tensor [3,H,W]
        instances = data_dict["instances"]       # detectron2.structures.Instances
        file_name = data_dict["file_name"]

        # from instances
        boxes = instances.gt_boxes.tensor
        labels = instances.gt_classes

        print(f"\n[{i}] 📸 file_name : {file_name}")
        print(f"🖼️ image shape : {tuple(image.shape)}")
        print(f"📦 boxes shape : {tuple(boxes.shape)}")
        print(f"🏷️ labels shape : {tuple(labels.shape)}")

        if i >= 9:
            break



