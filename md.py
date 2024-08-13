import torch
import numpy as np

# 假设这是你的矩阵
matrix = torch.tensor([[ 0.0000, 70.1503, 46.4004, 0],
                       [70.1503,  0.0000, 0, 96.4668],
                       [46.4004, 0,  0.0000, 0],
                       [0, 96.4668, 0,  0.0000]])

# 计算矩阵的最大值和最小值
max_val = torch.max(matrix)
min_val = torch.min(matrix)

# 归一化矩阵
normalized_matrix = (matrix - min_val) / (max_val - min_val)

print("归一化后的矩阵：")
print(normalized_matrix)
