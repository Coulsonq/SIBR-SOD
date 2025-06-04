import os
import numpy as np
import cv2
from scipy.ndimage import convolve


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    将图像归一化到 [0, 1] 范围。
    如果图像为常数图像（如全黑或全白），会在后面使用时排除 NaN。
    """
    mi, ma = image.min(), image.max()
    if abs(ma - mi) < 1e-7:
        return image.astype(np.float32)  # 避免除以 0
    return (image - mi) / (ma - mi)


def calculate_f_measure(
        gt: np.ndarray,
        pred: np.ndarray,
        beta2: float = 0.3
) -> float:
    """
    根据提供的逻辑计算单张图的 F-measure。
    其中阈值使用 2 * pred.mean() 的自适应方式。

    参数:
        gt (np.ndarray): 灰度的 Ground Truth, 值范围[0,1] 或 [0,255]。
        pred (np.ndarray): 灰度的预测结果, 值范围[0,1] 或 [0,255]。
        beta2 (float): F-measure 计算中 (1 + beta^2)，通常 beta^2=0.3。

    返回:
        float: 该图像的 F-measure。
    """
    # 转为浮点，并保证在 [0,1] 范围
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    if gt.max() > 1:
        gt /= 255.0
    if pred.max() > 1:
        pred /= 255.0

    # 防止 gt 全 0 的情况
    if gt.max() < 1e-7:
        # 如果 gt 全 0，那么预测全 0 的情况下应该算比较“好”，这里直接返回 1 或 0 亦可酌情处理
        # 这里返回 0 代表无法定义
        return 0.0

    # 自适应阈值
    threshold = 2.0 * pred.mean()

    label = (gt > 0.5).astype(np.uint8).flatten()  # 二值化
    sco_th = (pred.flatten() > threshold).astype(np.uint8)

    TP = np.sum((label == 1) & (sco_th == 1))
    FP = np.sum((label == 0) & (sco_th == 1))
    FN = np.sum((label == 1) & (sco_th == 0))

    # 这里的 beta2 = 0.3 即 (1 + 0.3^2)
    # 一般论文里 F_beta = ( (1+β^2)*Precision*Recall ) / (β^2*Precision + Recall )
    # 这里直接把 0.3 当作 (1 + beta^2) 也行，不过概念上稍有区别，维持与原代码一致即可
    recall = TP / (TP + FN + 1e-7)
    precision = TP / (TP + FP + 1e-7)
    f_measure = ((1 + beta2) * precision * recall) / (beta2 * precision + recall + 1e-7)

    return f_measure


def calculate_e_measure(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    计算 E-measure 的示例（与提问代码中一致），
    实际上这只是其某种简化公式，通常论文中的 E-measure 会更复杂。

    参数:
        gt (np.ndarray): 灰度的 Ground Truth, 值范围[0,1]。
        pred (np.ndarray): 灰度的预测结果, 值范围[0,1]。

    返回:
        float: E-measure。
    """
    # 归一化
    gt = normalize_image(gt)
    pred = normalize_image(pred)

    gt_mean = np.mean(gt)
    pred_mean = np.mean(pred)

    # 为了避免极端情况分母为0，如果都为常数，分母会变 0
    # 这里加一个小量来避免 NaN。
    numerator = 2.0 * (gt_mean * pred_mean)
    denominator = gt_mean ** 2 + pred_mean ** 2 + 1e-7

    e_measure = numerator / denominator
    return e_measure


def calculate_mae(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    计算单张图像的 Mean Absolute Error (MAE)。

    参数:
        gt (np.ndarray): 灰度的 Ground Truth, [0,1] 或 [0,255]。
        pred (np.ndarray): 灰度的预测结果, [0,1] 或 [0,255]。

    返回:
        float: MAE。
    """
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)
    if gt.max() > 1:
        gt /= 255.0
    if pred.max() > 1:
        pred /= 255.0

    return np.mean(np.abs(gt - pred))


def ssim(pred: np.ndarray, target: np.ndarray, C1=0.01 ** 2, C2=0.03 ** 2) -> float:
    """
    计算单通道图片的 SSIM（结构相似性指数）。
    """
    pred = pred.astype(np.float64)
    target = target.astype(np.float64)

    # 简易 3x3 均值卷积核
    kernel = np.ones((3, 3)) / 9.0

    mu_pred = convolve(pred, kernel, mode='constant')
    mu_target = convolve(target, kernel, mode='constant')

    sigma_pred = convolve(pred * pred, kernel, mode='constant') - mu_pred ** 2
    sigma_target = convolve(target * target, kernel, mode='constant') - mu_target ** 2
    sigma_pred_target = convolve(pred * target, kernel, mode='constant') - mu_pred * mu_target

    ssim_map = ((2.0 * mu_pred * mu_target + C1) * (2.0 * sigma_pred_target + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred + sigma_target + C2) + 1e-7)
    return ssim_map.mean()


def calculate_s_measure(pred: np.ndarray, target: np.ndarray) -> float:
    """
    计算 S-measure（结构相似性度量）。
    参照提问中的方法，利用 SSIM 的思想做前景、背景两部分的加权。
    """
    alpha = 0.5

    # 保证范围在 [0, 1]
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    if pred.max() > 1:
        pred /= 255.0
    if target.max() > 1:
        target /= 255.0

    fg_mean = np.mean(target)
    # 如果全黑或全白，特殊处理
    if fg_mean < 1e-7:
        # 全黑时，比较“背景”的相似性
        return ssim(1.0 - pred, 1.0 - target)
    elif abs(fg_mean - 1.0) < 1e-7:
        # 全白时，比较“前景”的相似性
        return ssim(pred, target)
    else:
        # 正常情况下，对前景、背景分别计算 SSIM，再做一个加权
        return alpha * ssim(pred, target) + (1 - alpha) * ssim(1.0 - pred, 1.0 - target)


def calculate_all_metrics(
        folder_pred: str,
        folder_gt: str,
        beta2: float = 0.3
):
    """
    同时计算文件夹中对应图像的 F-measure, E-measure, MAE, S-measure。
    该版本通过图像文件名匹配，而不是数量匹配。

    参数:
        folder_pred (str): 预测结果文件夹
        folder_gt (str):   真实标签文件夹
        beta2 (float):     F-measure 计算中使用的 beta2，默认为0.3。

    返回:
        dict: { 'F': [f1, f2, ...], 'E': [...], 'MAE': [...], 'S': [...] }
              以及四个指标的平均值 { 'F_mean': xx, 'E_mean': xx, ... }
    """
    pred_list = sorted(os.listdir(folder_pred))  # 获取预测文件夹中的文件列表
    gt_list = sorted(os.listdir(folder_gt))  # 获取真实标签文件夹中的文件列表

    F_scores = []
    E_scores = []
    MAEs = []
    S_scores = []

    for pred_name in pred_list:
        # 只处理与预测文件同名的真实图像
        gt_name = pred_name  # 假设文件名完全匹配

        # 构造文件路径
        pred_path = os.path.join(folder_pred, pred_name)
        gt_path = os.path.join(folder_gt, gt_name)

        if not os.path.exists(gt_path):
            print(f"Warning: {gt_path} 不存在，跳过 {pred_name}。")
            continue

        # 读入灰度图 (0~255)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            print(f"Warning: 图像 {pred_name} 或 {gt_name} 读取失败。")
            continue

        # 调整 pred 到与 gt 相同的尺寸
        h, w = gt_img.shape[:2]
        pred_img = cv2.resize(pred_img, (w, h), interpolation=cv2.INTER_LINEAR)

        # ---- 计算四个指标 ----
        f_val = calculate_f_measure(gt_img, pred_img, beta2)
        e_val = calculate_e_measure(gt_img, pred_img)
        mae_val = calculate_mae(gt_img, pred_img)
        s_val = calculate_s_measure(pred_img, gt_img)

        F_scores.append(f_val)
        E_scores.append(e_val)
        MAEs.append(mae_val)
        S_scores.append(s_val)

        print(f"Processed {pred_name}: F={f_val:.4f}, E={e_val:.4f}, MAE={mae_val:.4f}, S={s_val:.4f}")

    # 计算平均值
    f_mean = np.mean(F_scores) if F_scores else 0
    e_mean = np.mean(E_scores) if E_scores else 0
    mae_mean = np.mean(MAEs) if MAEs else 0
    s_mean = np.mean(S_scores) if S_scores else 0

    print("\n===== Final Results =====")
    print(f"F-measure: {f_mean:.4f}")
    print(f"E-measure: {e_mean:.4f}")
    print(f"MAE:       {mae_mean:.4f}")
    print(f"S-measure: {s_mean:.4f}")

    # 返回详细列表和平均值，方便后续使用
    return {
        'F': F_scores,
        'E': E_scores,
        'MAE': MAEs,
        'S': S_scores,
        'F_mean': f_mean,
        'E_mean': e_mean,
        'MAE_mean': mae_mean,
        'S_mean': s_mean
    }
if __name__ == "__main__":
    # 示例：修改为自己对应的文件夹路径
    path_pred = r"D:\object\mycode\pred"
    path_gt = r'C:\Users\A\Desktop\SOD_dataset\DUT_TE\gt'

    results = calculate_all_metrics(path_pred, path_gt, beta2=0.3)
