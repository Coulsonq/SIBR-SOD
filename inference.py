import shutil

import torch
from torch.utils.data import DataLoader
from dataset import SaliencyDataset
from network import Salmodel
import os
from PIL import Image
import numpy as np
from metrics import calculate_all_metrics

def predict(img_dir, model_path, save_dir='predictions', batch_size=16):
    """
    加载模型权重并进行预测，将预测结果保存为 PNG 图片
    """
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    dataset = SaliencyDataset(img_dir, mode='test')  # 不需要 GT 数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = Salmodel().to(device)
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()  # 设置为评估模式

    # 创建保存预测结果的目录
    os.makedirs(save_dir, exist_ok=True)

    # 预测循环
    with torch.no_grad():  # 禁用梯度计算
        for i, (img, img_name_list) in enumerate(dataloader):
            img = img.to(device)

            # 前向传播
            mask = model(img)

            # 将预测结果转换为二值图像
            pred_mask = torch.sigmoid(mask)  # 使用 sigmoid 将 logits 转换为概率
            #pred_mask = (pred_mask > 0.5).float()  # 二值化

            # 将预测结果保存为 PNG 图片
            for j in range(pred_mask.shape[0]):
                mask_np = pred_mask[j].squeeze().cpu().numpy()  # 转换为 numpy 数组
                mask_np = (mask_np * 255).astype(np.uint8)  # 转换为 0-255 的灰度值
                mask_img = Image.fromarray(mask_np)  # 转换为 PIL 图像
                img_name = img_name_list[j]
                # 保存图片
                save_path = os.path.join(save_dir, "{}.png".format(img_name))
                mask_img.save(save_path)
                print(f"预测结果已保存到 {save_path}")

    print("预测完成！")

if __name__ == '__main__':
    model_path = r"D:\object\mycode\checkpoints\model_epoch_51.pth"  # 训练好的模型权重路径
    img_dir = r'D:\object\mycode\test\imgs'
    save_dir = r"D:\object\mycode\test_pred"  # 保存预测结果的目录
    path_gt = r'C:\Users\A\Desktop\duts-te\GT'

    shutil.rmtree(save_dir)  # 删除整个文件夹
    os.makedirs(save_dir)
    predict(img_dir, model_path, save_dir)
    #results = calculate_all_metrics(save_dir, path_gt, beta2=0.3)

