import torch
import math
from torch import nn
import numpy as np

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResNormBlock(nn.Module):
    def __init__(self, module, d_model=256, dropout=0.1):
        super().__init__()
        self.module = module
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, *args, **kwargs):
        out = self.module(x, *args, **kwargs)
        return self.norm(x + self.dropout(out))

class DynamicConv(nn.Module):

    def __init__(self):
        super().__init__()

        self.hidden_dim = 256
        self.dim_dynamic = 64
        self.num_dynamic = 2
        self.num_params = self.hidden_dim * self.dim_dynamic
        self.dynamic_layer = nn.Linear(self.hidden_dim, self.num_dynamic * self.num_params)

        self.norm1 = nn.LayerNorm(self.dim_dynamic)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        self.activation = nn.ReLU(inplace=True)

        pooler_resolution = 7
        num_output = self.hidden_dim * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)

    def forward(self, pro_features, roi_features):
        '''
        pro_features: (1,  N * nr_boxes, self.d_model)
        roi_features: (49, N * nr_boxes, self.d_model)

        ret : [1, 2000, 256]
        '''
        features = roi_features.permute(1, 0, 2)
        parameters = self.dynamic_layer(pro_features).permute(1, 0, 2)

        param1 = parameters[:, :, :self.num_params].view(-1, self.hidden_dim, self.dim_dynamic)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dim_dynamic, self.hidden_dim)

        features = torch.bmm(features, param1) # 배치별 행렬곱 
        features = self.norm1(features)
        features = self.activation(features)

        features = torch.bmm(features, param2)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)
        features = self.out_layer(features)
        features = self.norm3(features)
        features = self.activation(features)

        return features

class SABlock(nn.Module):
    """
    Self-Attention Block using ResNormBlock
    input  : [B, N, C]
    output : [B, N, C]
    # nhead 8 인 self-attention 
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
        )

        # ResNormBlock 안에 self-attn module 람다로 wrapping
        self.block = ResNormBlock(
            module=lambda x: self.self_attn(x, x, x)[0],
            d_model=d_model,
            dropout=dropout
        )

    def forward(self, x):
        """
        
        """
        # [B, N, C] --> [N, B, C]
        x = x.permute(1, 0, 2)
        return self.block(x)

class DCBlock(nn.Module):
    """
    DynamicConv + Residual + LayerNorm block
    pro_features shape: [1, N*nr_boxes, 256]
    roi_features shape: [49, N*nr_boxes, 256]
    """

    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dynamic_conv = DynamicConv()

        # module: lambda를 사용해 pro_features를 x로 취급하고 roi_features는 추가 인자로 전달
        self.block = ResNormBlock(
            module=lambda x, roi: self.dynamic_conv(x, roi),
            d_model=d_model,
            dropout=dropout
        )

    def forward(self, pro_f, roi_f):
        """
        Residual path = pro_features
        DynamicConv output shape must match pro_features (256 dim)
        """
        # pro_f [N  , B,  C]
        # roi_f [BxN, C, 49]

        pro_f = pro_f.permute(1, 0, 2).reshape(1, -1, self.d_model) # [N, B, C]    -->     [ 1, B x N, C]
        roi_f = roi_f.permute(2, 0, 1)                              # [B x N, C, 49] -->   [49, B x N, C]

        return self.block(pro_f, roi_f)


class MLPBlock(nn.Module):
    """
    Feed-Forward Network (FFN) Block using ResNormBlock
    input  : [B, N, C]
    output : [B, N, C]
    """

    def __init__(self, d_model=256, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # MLP: Linear -> ReLU -> Linear
        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, d_model),
        )

        # ResNormBlock wrapping
        self.block = ResNormBlock(
            module=lambda x: self.mlp(x),
            d_model=d_model,
            dropout=dropout
        )

    def forward(self, x):
        return self.block(x)


class TimeAdaINBlock(nn.Module):
    """
    AdaIN-style modulation using time embedding.
    Produces scale & shift from time_emb and modulates features.
    """

    def __init__(self, d_model=256):
        super().__init__()

        # time_emb: [B, d_model*4] -> [B, d_model*2] (scale + shift)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 2)
        )

    def forward(self, x, time_emb, nr_boxes):
        """
        x        : [1, B x N, C]
        time_emb : [B, 512]
        """
        #
        x = x[0]                                                             # [B * N, C]
        scale_shift = self.time_mlp(time_emb)                                # [B, 512]

        # repeat for each query box
        scale_shift = torch.repeat_interleave(scale_shift, nr_boxes, dim=0)  # [B*N, 512]
        scale, shift = scale_shift.chunk(2, dim=1)

        # AdaIN-style modulation: (scale + 1) * x + shift
        return x * (scale + 1) + shift
    

class ClsHead(nn.Module):
    """
    Classification head:
    Linear → LayerNorm → ReLU
    """
    def __init__(self, d_model=256):
        super().__init__()

        self.fc = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.act(x)
        return x
    


class BoxHead(nn.Module):
    """
    Regression head:
    (Linear → LayerNorm → ReLU) × 3
    """
    def __init__(self, d_model=256):
        super().__init__()

        self.fc1  = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.act1  = nn.ReLU(inplace=True)

        self.fc2  = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.act2  = nn.ReLU(inplace=True)

        self.fc3  = nn.Linear(d_model, d_model, bias=False)
        self.norm3 = nn.LayerNorm(d_model)
        self.act3  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act1(self.norm1(self.fc1(x)))
        x = self.act2(self.norm2(self.fc2(x)))
        x = self.act3(self.norm3(self.fc3(x)))
        return x
