# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from common import LayerNorm2d
from torchvision.ops import DeformConv2d


import torch
import torch.nn as nn

class TransformerFeatureUpsample(nn.Module):
    def __init__(self, in_channels=256, out_channels=32):
        super().__init__()

        # 第一次上采样：14→28
        self.conv1 = nn.Conv2d(in_channels, 128 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)

        # 第二次上采样：28→56
        self.conv2 = nn.Conv2d(128, 64 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)

        # 第三次上采样：56→112
        self.conv3 = nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1)
        self.pixel_shuffle3 = nn.PixelShuffle(2)

        # 第四次上采样：112→224
        self.conv4 = nn.Conv2d(64, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle4 = nn.PixelShuffle(2)

        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.shape  # 输入：(B,256,14,14)

        # 第一次上采样 (14→28)
        x = self.conv1(x)               # (B,512,14,14)
        x = self.pixel_shuffle1(x)      # (B,128,28,28)
        x = self.act(x)

        # 第二次上采样 (28→56)
        x = self.conv2(x)               # (B,256,28,28)
        x = self.pixel_shuffle2(x)      # (B,64,56,56)
        x = self.act(x)

        # 第三次上采样 (56→112)
        x = self.conv3(x)               # (B,256,56,56)
        x = self.pixel_shuffle3(x)      # (B,64,112,112)
        x = self.act(x)

        # 第四次上采样 (112→224)
        x = self.conv4(x)               # (B,128,112,112)
        x = self.pixel_shuffle4(x)      # (B,32,224,224)
        x = self.act(x)

        return x


class DeformableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConv, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)  # 计算偏移量
        x = self.deform_conv(x, offset)  # 进行可变形卷积
        x = self.norm(x)  # 归一化
        return self.act(x)

class ReduceToOneChannel(nn.Module):
    def __init__(self):
        super(ReduceToOneChannel, self).__init__()

    def forward(self, x):
        return self.layers(x)

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer1: nn.Module,
        transformer2: nn.Module,
        transformer3: nn.Module,
        transformer4: nn.Module,
        activation: Type[nn.Module] = nn.GELU,
        norm : Type[nn.Module] = nn.BatchNorm2d,
        act : Type[nn.Module] = nn.GELU
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer1 = transformer1
        self.transformer2 = transformer2
        self.transformer3 = transformer3
        self.transformer4 = transformer4

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(1, transformer_dim)

        '''
        self.output_upscaling_deconv = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=5, stride=4, padding=1, output_padding=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=5, stride=4, padding=1, output_padding=1),
            activation(),
        )

        
        self.output_upscaling_bi = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1),
            activation(),
        )
        '''

        dim = 32


        self.conv_fuse = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(256),
                                       nn.LeakyReLU(inplace=True))


        self.sigmoid = nn.Sigmoid()
        self.conv_p1 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True)
        self.conv_matt = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
                                       nn.BatchNorm2d(32),
                                       nn.LeakyReLU(inplace=True))
        #self.no_mask_embed = nn.Embedding(1, 256)

        self.output_hypernetworks_mlps = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.upsample_module = TransformerFeatureUpsample(in_channels=256, out_channels=32)
        self.fus_conv = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        #global_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
          :param global_feature:
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            #global_feature=global_feature,
        )

        # Select the correct mask or masks for output

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        #global_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(image_embeddings[-1].shape[0],-1,-1)

        #src = torch.cat([image_embeddings, global_feature], dim=1)# + prompt_token# + prompt_token
        #src = self.conv_fuse(src)
        src1, src2, src3, src4 = image_embeddings
        pos_src = image_pe
        b, c, h, w = src4.shape

        # Run the transformer
        hs1, src1 = self.transformer1(src1, pos_src, output_tokens)
        hs2, src2 = self.transformer2(src2, pos_src, output_tokens)
        hs3, src3 = self.transformer3(src3, pos_src, output_tokens)
        hs4, src4 = self.transformer4(src4, pos_src, output_tokens)

        hs = output_tokens[:,1:,:]

        hs = self.output_hypernetworks_mlps(hs)

        # Upscale mask embeddings and predict masks using the mask tokens
        src_up1 = src1.transpose(1, 2).view(b, c, h, w).contiguous()
        src_up2 = src2.transpose(1, 2).view(b, c, h, w).contiguous()
        src_up3 = src3.transpose(1, 2).view(b, c, h, w).contiguous()
        src_up4 = src4.transpose(1, 2).view(b, c, h, w).contiguous()
        combined_src_up = torch.cat([src_up1, src_up2, src_up3, src_up4], dim=1)
        combined_src_up = self.fus_conv(combined_src_up)

        upscaled_embedding = self.upsample_module(combined_src_up)
        #upscaled_embedding = self.output_upscaling_deconv(combined_src_up)
        #upscaled_embedding = self.output_upscaling_bi(combined_src_up)

        b, c, h, w = upscaled_embedding.shape

        coarse_mask = (hs @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        #coarse_mask = F.interpolate(coarse_mask, [224, 224], mode='bilinear',
                                                      #align_corners=False)

        return coarse_mask


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
