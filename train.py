import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import SaliencyDataset
from network import Salmodel  # 确保 `network.py` 里 `model` 是 `torch.nn.Module` 实例
from tqdm import tqdm  # 导入 tqdm 库
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import LossFunc

import torch
import torch.nn as nn
import torch.nn.functional as F


def reshapePos(pos_embed, img_size):
    token_size = int(img_size // 16)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not ('2' in k or '5' in k or '8' in k or '11' in k):
        return rel_pos_params

    token_size = int(img_size // 16)
    h, w = rel_pos_params.shape
    rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
    rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
    return rel_pos_params[0, 0, ...]

def load(net,ckpt, img_size):
    ckpt=torch.load(ckpt,map_location='cpu')
    from collections import OrderedDict
    dict=OrderedDict()
    for k,v in ckpt.items():
        #把pe_layer改名
        if 'pe_layer' in k:
            dict[k[15:]] = v
            continue
        if 'pos_embed' in k :
            dict[k] = reshapePos(v, img_size)
            continue
        if 'rel_pos' in k:
            dict[k] = reshapeRel(k, v, img_size)
        elif "image_encoder" in k:
            if "neck" in k:
                #Add the original final neck layer to 3, 6, and 9, initialization is the same.
                for i in range(4):
                    new_key = "{}.{}{}".format(k[:18], i, k[18:])
                    dict[new_key] = v
            else:
                dict[k]=v
        if "mask_decoder.transformer" in k:
            dict[k] = v
        if "mask_decoder.iou_token" in k:
            dict[k] = v
        if "mask_decoder.output_upscaling" in k:
            dict[k] = v
    state = net.load_state_dict(dict, strict=False)
    return state

def train(img_dir, gt_dir, epochs=10, batch_size=16, lr=None, model_path=None):
    """
    训练显著性检测模型
    """
    # 设备选择
    save_dir = 'checkpoints'

    device = torch.device('cuda')

    # 加载数据
    dataset = SaliencyDataset(img_dir, gt_dir, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = Salmodel()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    #else:
        #state = load(model, SAM_weight, img_size=224)
        #model.load_state_dict(torch.load(SAM_weight), strict=False)

    model.to(device)
    # 损失函数 & 优化器

    #criterion = nn.BCEWithLogitsLoss()
    criterion = LossFunc

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 创建保存权重的目录
    os.makedirs(save_dir, exist_ok=True)

    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # 使用 tqdm 显示进度条
        with tqdm(dataloader, desc=f'Epoch [{epoch+1}/{epochs}]', unit='batch') as tepoch:
            for img, gt, img_name_list in tepoch:
                img, gt = img.to(device), gt.to(device)

                # 前向传播
                #mask_1, mask_2, mask_3, mask_4 = model(img)
                mask_1 = model(img)
                mask_1 = torch.sigmoid(mask_1)
                #mask_2 = torch.sigmoid(mask_2)
                #mask_3 = torch.sigmoid(mask_3)
                #mask_4 = torch.sigmoid(mask_4)

                loss1 = criterion(mask_1, gt)
                #loss2 = criterion(mask_2, gt)
                #loss3 = criterion(mask_3, gt)
                #loss4 = criterion(mask_4, gt)

                #loss = 0.2 * loss1 + 0.2 * loss2 + 0.2 * loss3 + 0.6 * loss4
                loss = loss1

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # 保存模型权重
        checkpoint_path = os.path.join(r'D:\object\mycode\deconv_chkp', f'model_epoch_{epoch+1}.pth')
        if epoch % 3 == 0 :
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型权重已保存到 {checkpoint_path}")

    print("训练完成！")

if __name__ == '__main__':
    img_dir = r"C:\Users\A\Desktop\all\2d_dataset2\imgs"
    gt_dir = r"C:\Users\A\Desktop\all\2d_dataset2\labels"

    #model_path = r'D:\object\mycode\checkpoints\base.pth'
    model_path = None

    train(img_dir, gt_dir, epochs=50, batch_size=64, lr=1e-4, model_path=model_path)