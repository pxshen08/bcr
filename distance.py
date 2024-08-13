import torch
import numpy as np
# 计算欧几里得距离函数
def euclidean_distance(x1, x2):

    return torch.norm(x1 - x2, p=2, dim=0)
def mahalanobis_distance(x, y, cov):
    """
    计算两个向量之间的马哈拉诺比斯距离。

    参数：
    x: 第一个向量，形状为 (n_features,)
    y: 第二个向量，形状为 (n_features,)
    cov: 数据的协方差矩阵，形状为 (n_features, n_features)

    返回值：
    马哈拉诺比斯距离
    """
    diff = x - y
    inv_cov = np.linalg.inv(cov)
    distance = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return distance
def covariance(x, y):
    """
    计算两个向量之间的协方差。
    参数：
    x: 第一个向量，形状为 (n_features,)
    y: 第二个向量，形状为 (n_features,)
    返回值：
    协方差
    """
    n = x.size(0)  # 假设 x 和 y 都有相同的长度
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    cov = torch.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return cov.item()