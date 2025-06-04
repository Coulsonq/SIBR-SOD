import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 允许重复加载OpenMP库（临时解决方案）
# 读取灰度图
gray_img = cv2.imread(r'D:\object\mycode\pred\0982.png', cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    raise FileNotFoundError("Image not found at the specified path.")

# 将NumPy数组转换为PyTorch Tensor，并添加batch和channel维度 [1, 1, H, W]
gray_tensor = torch.from_numpy(gray_img).float().unsqueeze(0).unsqueeze(0)

# 平均池化（2x2，步长2）
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
pooled_tensor = avg_pool(gray_tensor)

# 将池化结果转回NumPy数组，并移除多余的维度 [H/2, W/2]
pooled_img = pooled_tensor.squeeze().numpy()

# 上采样池化结果到原图尺寸
upsampled_pooled = cv2.resize(pooled_img, (gray_img.shape[1], gray_img.shape[0]),
                             interpolation=cv2.INTER_LINEAR)

# 计算残差（高频细节）
residual = np.abs(gray_img.astype(np.float32) - upsampled_pooled.astype(np.float32))
residual_uint8 = residual.astype(np.uint8)  # 直接转为uint8（可能截断）

# 可视化
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(upsampled_pooled, cmap='gray')
plt.title('Upsampled Pooled Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(residual_uint8, cmap='gray')
plt.title('Residual (High-Frequency Details)')
plt.axis('off')

plt.show()