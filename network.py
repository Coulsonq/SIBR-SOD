import torch
from torch import nn
from SAMencoder import ImageEncoderViT
from SAMdecoder import MaskDecoder
from dataset import SaliencyDataset
from torch.utils.data import Dataset, DataLoader
from typing import Any, Optional, Tuple, Type
import numpy as np
from towaytransformer import TwoWayTransformer
from PIL import Image
import torch.nn.functional as F
from CRM import CRM
from block import DetailEnhancement

size = 224
class MLFusion(nn.Module):
    def __init__(self, norm=nn.BatchNorm2d, act=nn.ReLU):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            norm(256),
            act(),
        )

        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(256, 256, 1, bias=False),
                norm(256),
                act(),
            ))

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature_list):
        fusi_feature = torch.cat(feature_list, dim=1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)

        for i in range(4):
            x = feature_list[i]
            attn = self.attn_conv[i](x)
            attn = self.pool(attn)
            attn = self.sigmoid(attn)

            x = attn * x + x
            feature_list[i] = x

        return feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]

class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

pe_layer = PositionEmbeddingRandom(256 // 2)
get_dense_pe = pe_layer([size // 16, size // 16]).unsqueeze(0)
img_pe = get_dense_pe.to('cuda')

class global_fusion(nn.Module):
    def __init__(self, channel_dim):
        super(global_fusion, self).__init__()
        self.fus = nn.Sequential(
            nn.Conv2d(in_channels=channel_dim, out_channels=channel_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=channel_dim // 2, out_channels=channel_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim // 4),
            nn.ReLU()
        )
    def forward(self, x):
        return self.fus(x)

def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        还原显著性掩码的尺寸：
        - 先去除 padding
        - 再放大到原始图像大小
        """
        masks = F.interpolate(
            masks,
            (self.encoder.img_size, self.encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 裁剪掉 padding
        masks = masks[..., : input_size[0], : input_size[1]]

        # 上采样到原始图像大小
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

        return masks

class Salmodel(nn.Module):
    def __init__(self):
        super(Salmodel, self).__init__()
        self.encoder = ImageEncoderViT()
        self.decoder1 = MaskDecoder(transformer1=TwoWayTransformer(
                                        depth=2,
                                        embedding_dim=256,
                                        mlp_dim=2048,
                                        num_heads=8
                                        ),
            transformer2=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            ),
            transformer3=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            ),
            transformer4=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8
            ),
                transformer_dim=256,
                norm=nn.BatchNorm2d,
                act=nn.ReLU)

        self.fus = MLFusion()

        self.deep_feautre_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.detail_enhance = DetailEnhancement(img_dim=32, feature_dim=32, norm=nn.BatchNorm2d, act=nn.ReLU)

    def forward(self, img):
        feature_list = self.encoder(img)
        last_feature = feature_list[-1]
        #globael_feature_dec = self.fus(img_feature_list)
        #coarse_mask, feature_dec = self.decoder(last_feature, img_pe, globael_feature_dec)
        mask = self.decoder1(feature_list, img_pe)

        #deep_feature = self.deep_feautre_conv(last_feature)
        #fine_mask = self.detail_enhance(img, feature_dec, deep_feature)


        return mask






