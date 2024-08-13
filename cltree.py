import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import pandas as pd
from distree import DisTree


# 定义神经网络模型
class BCRNet(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(BCRNet, self).__init__()
        # Task A 输出层
        self.fc_A = nn.Linear(embedding_dim, num_classes)
        # Task B 输出层
        self.fc_B = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        # Task A 输出
        output_A = nn.functional.softmax(self.fc_A(x), dim=1)
        # Task B 输出
        output_B = self.fc_B(x)
        return output_A, output_B

# 计算 Euclidean distance
def euclidean_distance(x1, x2):
    return torch.norm(x1 - x2, p=2, dim=1)


# 模型参数
embedding_dim = 100
num_classes = 3  # 假设有3个类别
lr = 0.001
alpha = 0.5  # Task A 权重
beta = 0.5   # Task B 权重

# 创建模型
model = BCRNet(embedding_dim, num_classes)
criterion_A = nn.CrossEntropyLoss()  # Task A 的损失函数
criterion_B = nn.MSELoss()  # Task B 的损失函数
optimizer = optim.Adam(model.parameters(), lr=lr)

# 模拟数据
num_samples = 1000
embedding_data = np.random.randn(num_samples, embedding_dim)
# 模拟 Task A 的标签
label_data_A = np.random.randint(0, num_classes, size=(num_samples,))
# 模拟 Task B 的树上距离
tree_distance_data = np.random.uniform(low=0.1, high=1.0, size=(num_samples,))

# 训练模型
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    optimizer.zero_grad()
    # 随机抽取一组 BCR 数据
    indices = np.random.choice(num_samples, batch_size, replace=False)
    x_batch = torch.Tensor(embedding_data[indices])
    label_batch_A = torch.LongTensor(label_data_A[indices])
    tree_distance_batch = torch.Tensor(tree_distance_data[indices])
    # 前向传播
    output_A, output_B = model(x_batch)
    # 计算 Task A 的损失
    loss_A = criterion_A(output_A, label_batch_A)
    # 计算 Euclidean distance
    euclidean_dist_batch = euclidean_distance(x_batch[0], x_batch[1])
    # 计算 Task B 的损失
    loss_B = criterion_B(output_B.squeeze(), tree_distance_batch)
    # 总损失
    loss_total = alpha * loss_A + beta * loss_B
    # 反向传播及优化
    loss_total.backward()
    optimizer.step()

    # 打印损失
    print(f'Epoch [{epoch+1}/{num_epochs}], Task A Loss: {loss_A.item():.4f}, Task B Loss: {loss_B.item():.4f}')

# 模型保存
torch.save(model.state_dict(), 'bcr_model_taskAB.pth')

if __name__ == "__main__":
    # 示例用法
    filename = "/home/mist/ClonalTree/Examples/output1/Kaminski_TS2.abRT.nk.csv"
    tree_graph = pd.read_csv(filename, header=None, names=['source', 'target', 'weight'])
    tree = tree_graph.values.tolist()
    print(tree)
    # 计算两个BCR在树上的距离
    BCR1 = 'seq468330'  # source_node
    BCR2 = 'seq468234'  # target_node
    distance = DisTree().sumOfDistancesInTree(tree, BCR1, BCR2)
    print(f"Distance between BCR1 and BCR2: {distance}")