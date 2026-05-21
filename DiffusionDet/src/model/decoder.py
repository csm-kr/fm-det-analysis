import math
import torch
import torch.nn as nn
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures.boxes import Boxes
from src.model.blocks import SinusoidalPositionEmbeddings, SABlock, DCBlock, MLPBlock, TimeAdaINBlock, ClsHead, BoxHead

class Decoder(nn.Module):

    def __init__(self, d_model=256):
        super().__init__()

        self.return_intermediate = True
        self.num_classes = 80
        
        # 256 --> 1024
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, 4 * d_model), 
        )

        # set 'roi pooler'
        self.roi_pooler = ROIPooler(
            output_size=7,
            scales=(0.25, 0.125, 0.0625, 0.03125),
            sampling_ratio=2,
            pooler_type='ROIAlignV2',
        )

        # num_heads 6 
        self.heads = nn.ModuleList([DetectionHead(d_model=256, num_classes=80) for i in range(6)]) 

        # focal loss
        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if p.shape[-1] == self.num_classes or p.shape[-1] == self.num_classes + 1:
                nn.init.constant_(p, self.bias_value)
                # print(p) -4.5951
    
    def forward(self, features_list, init_bboxes, t, is_eval=False):

        pred_logits_list = []
        pred_bboxes_list = []

        # get time embed
        time_emb = self.time_embed(t)
        proposal_features = None
        bboxes = init_bboxes

        for head in self.heads:
            pred_logits, pred_bboxes, proposal_features = head(features_list, bboxes, self.roi_pooler, time_emb, proposal_features)

            pred_logits_list.append(pred_logits)
            pred_bboxes_list.append(pred_bboxes)
            bboxes = pred_bboxes.detach() # update bboxes

        if is_eval:
            return torch.stack(pred_logits_list)[-1], torch.stack(pred_bboxes_list)[-1]

        pred_logits_tensor = torch.stack(pred_logits_list).permute(1, 0, 2, 3)  # [B, num_heads, H, W]
        pred_bboxes_tensor = torch.stack(pred_bboxes_list).permute(1, 0, 2, 3)  # [B, num_heads, 4]

        return pred_logits_tensor, pred_bboxes_tensor

class DetectionHead(nn.Module):
    def __init__(self, d_model=256, num_classes=80):
        super().__init__()

        # ----------------------------------------------------------
        # 예시: Self-Attention 모듈
        # ----------------------------------------------------------
        self.d_model = d_model
        self.num_classes = num_classes

        self.sa_block = SABlock()
        self.dc_block = DCBlock()
        self.mlp_block = MLPBlock()
        self.time_adain = TimeAdaINBlock()
        self.cls_head = ClsHead()
        self.box_head = BoxHead()

        self.scale_clamp = math.log(100000.0 / 16)
        self.cls_logits = nn.Linear(d_model, num_classes)
        self.box_delta = nn.Linear(d_model, 4)
        self.bbox_weights = (2.0, 2.0, 1.0, 1.0)


    def forward(self, features_list, bboxes, roi_pooler, time_emb, pro_f=None):
        """
        Args:
            features_list            : list of feature maps (FPN output)
            bboxes     : initial proposal boxes [B, N, 4]
            roi_pooler : ROI pooling module
            time_emb   : diffusion timestep embedding [B, d_model*4]
            pro_f      : previous proposal features (optional)

        Returns:
            pred_logits : class logits for each proposal   [B, N, num_classes]
            pred_bboxes : predicted bounding boxes         [B, N, 4]
            pro_f       : refined proposal features        [B, N, C]
        """
        # get batch size, number_of_init_boxes
        B, N = bboxes.shape[:2]

        proposal_boxes = []
        B = bboxes.shape[0]

        for i in range(B):
            proposal_boxes.append(Boxes(bboxes[i]))                             # List[Boxes]

        # roi_f = roi_pooler(features_list, bboxes)                             # [B x N, C, 7, 7]
        roi_f = roi_pooler(features_list, proposal_boxes)                       # OK

        roi_f = roi_f.flatten(-2)                                               # [B x N, C, 49]
        if pro_f is None:
            pro_f = roi_f.view(B, N, self.d_model, -1).mean(-1)                 # [B, N, C, 49] --> [B, N, C]

        # self-attn    res-norm block 
        pro_f = self.sa_block(pro_f)                                            # [B, N, C] --> [N, B, C]
        # dynamic conv res-norm block 
        pro_f = self.dc_block(pro_f, roi_f)                                     # [1, B x N, C]
        # mlp          res-norm block 
        pro_f = self.mlp_block(pro_f)                                           # [1, B x N, C]                   [1, 2000, 256]     

        # time adain 
        cond_f = self.time_adain(pro_f, time_emb, N)                            # [B x N, C]

        cls_f = cond_f.clone()
        box_f = cond_f.clone()
        # cls & reg heads
        cls_f = self.cls_head(cls_f)                                            # [B x N, C]
        box_f = self.box_head(box_f)                                            # [B x N, C]

        # predict cls
        pred_logits = self.cls_logits(cls_f)                                    # [B x N, 80]

        # predict boxes
        bboxes_deltas = self.box_delta(box_f)                                   # [B x N, 4]
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1, 4))      # [B x N, 4]
        
        # reshape pro_f
        pred_logits = pred_logits.view(B, N, -1)                                # [B, N, 80]
        pred_bboxes = pred_bboxes.view(B, N, -1)                                # [B, N, 4]
        pro_f = pro_f.view(B, N, self.d_model)                                  # [B, N, C]

        return pred_logits, pred_bboxes, pro_f
    
    
    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4), where k >= 1.
                deltas[i] represents k potentially different class-specific
                box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        boxes = boxes.to(deltas.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.bbox_weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w  # x1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h  # y1
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w  # x2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h  # y2

        return pred_boxes
