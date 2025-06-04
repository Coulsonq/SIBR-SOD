import torch
import torch.nn.functional as F
import torch.nn as nn


# **BCE Loss + IoU Loss + Edge Loss + Gradient Loss**
def LossFunc(pred, mask, alpha=1.0, beta=1.0, gamma=2.0, delta=2.0):
    """
    强化边缘清晰度的综合损失：
    - BCE Loss (二元交叉熵) 保证基本的显著性区域预测
    - IoU Loss (交并比) 让预测区域与 GT 更匹配
    - Edge Loss (梯度损失) 关注 X/Y 方向的边缘信息
    - Gradient Loss (Sobel 边缘损失) 保证锐利边界
    """

    # **1. BCE Loss**
    bce_loss = F.binary_cross_entropy(pred, mask, reduction='mean')

    # **2. IoU Loss**
    intersection = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3)) - intersection
    iou_loss = 1 - (intersection + 1) / (union + 1)  # 避免除零
    iou_loss = iou_loss.mean()

    # **3. Edge Loss (基于 X/Y 方向梯度差异)**
    pred_x = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    pred_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    mask_x = torch.abs(mask[:, :, 1:, :] - mask[:, :, :-1, :])
    mask_y = torch.abs(mask[:, :, :, 1:] - mask[:, :, :, :-1])

    # 统一 shape，避免 shape mismatch
    pred_x = F.pad(pred_x, (0, 0, 0, 1))  # pad height
    pred_y = F.pad(pred_y, (0, 1, 0, 0))  # pad width
    mask_x = F.pad(mask_x, (0, 0, 0, 1))
    mask_y = F.pad(mask_y, (0, 1, 0, 0))

    edge_loss = F.l1_loss(pred_x + pred_y, mask_x + mask_y, reduction='mean')

    # **4. Sobel Gradient Loss (梯度对齐)**
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).to(pred.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).to(pred.device).view(1, 1, 3, 3)

    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    mask_grad_x = F.conv2d(mask, sobel_x, padding=1)
    mask_grad_y = F.conv2d(mask, sobel_y, padding=1)

    gradient_loss = F.l1_loss(pred_grad_x, mask_grad_x, reduction='mean') + F.l1_loss(pred_grad_y, mask_grad_y,
                                                                                      reduction='mean')

    # **最终 Loss**
    total_loss = alpha * bce_loss + beta * iou_loss + gamma * edge_loss + delta * gradient_loss
    return total_loss