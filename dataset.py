import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

size = 224
class SaliencyDataset(Dataset):
    def __init__(self, rgb_dir, gray_dir=None, mode='train'):
        """
        显著性检测数据集
        :param rgb_dir: RGB 图像目录
        :param gray_dir: 灰度图像（显著性图）目录
        :param mode: 训练模式 'train' 或 测试模式 'test'
        """
        self.rgb_dir = rgb_dir
        self.gray_dir = gray_dir
        self.mode = mode  # 控制不同模式的数据变换

        # 训练模式（归一化 + 数据增强）
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),  # 统一大小
                #transforms.RandomHorizontalFlip(p=0.5),  # 50% 概率水平翻转
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),  # 颜色抖动
                transforms.ToTensor(),  # 转换为张量
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
            ])
        # 测试模式（仅缩放和张量转换）
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        # 处理灰度图（显著性图）
        self.target_transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

        # 获取所有 RGB 图像的文件名
        self.filenames = sorted(os.listdir(rgb_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 获取文件名（不包括扩展名）
        filename = os.path.splitext(self.filenames[idx])[0]

        # 读取 RGB 图像（.jpg）
        rgb_path = os.path.join(self.rgb_dir, filename + ".jpg")
        rgb_image = Image.open(rgb_path).convert("RGB")
        # 应用变换
        if self.transform:
            rgb_image = self.transform(rgb_image)

        if self.gray_dir is not None:
            # 读取灰度图（.png）
            gray_path = os.path.join(self.gray_dir, filename + ".png")
            gray_image = Image.open(gray_path).convert("L")  # 单通道灰度图
            if self.target_transform:
                gray_image = self.target_transform(gray_image)

        if self.gray_dir is not None:
            return rgb_image, gray_image, filename
        else:
            return rgb_image, filename

if __name__ == "__main__":
    # 数据集路径
    rgb_dir = "/Users/coulsonq/Documents/mycode/pics/img"
    gray_dir = "/Users/coulsonq/Documents/mycode/pics/gt"

    # 训练数据集
    train_dataset = SaliencyDataset(rgb_dir, gray_dir, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 测试数据集
    test_dataset = SaliencyDataset(rgb_dir, gray_dir, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 训练数据加载测试
    for rgb, gray, filename in train_dataloader:
        print("【训练模式】")
        print("RGB 形状:", rgb.shape)  # [batch_size, 3, 224, 224]
        print("灰度图形状:", gray.shape)  # [batch_size, 1, 224, 224]
        print("文件名:", filename)
        break

    # 测试数据加载测试
    for rgb, gray, filename in test_dataloader:
        print("【测试模式】")
        print("RGB 形状:", rgb.shape)  # [batch_size, 3, 224, 224]
        print("灰度图形状:", gray.shape)  # [batch_size, 1, 224, 224]
        print("文件名:", filename)
        break