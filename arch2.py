import torch
import torch.nn as nn
import torch.nn.functional as F

# Instance Embedding
class InstanceEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InstanceEmbedding, self).__init__()
        self.attention = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        attention_weights = F.softmax(self.attention(x), dim=1)
        instance_embedding = torch.sum(attention_weights * x, dim=1)
        return instance_embedding

# Bag Classification
class BagClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(BagClassification, self).__init__()
        self.gated_attention = nn.Linear(input_dim, 1)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attention_weights = F.sigmoid(self.gated_attention(x))
        pooled_representation = torch.sum(attention_weights * x, dim=1)
        out = F.relu(self.fc1(pooled_representation))
        out = self.fc2(out)
        return out

from torch.utils.data import Dataset, DataLoader
import numpy as np

class MILDataset(Dataset):
    def __init__(self, num_bags, instances_per_bag, input_dim):
        self.num_bags = num_bags
        self.instances_per_bag = instances_per_bag
        self.input_dim = input_dim

        self.data = []
        self.labels = []

        for _ in range(num_bags):
            bag_data = torch.randn(instances_per_bag, input_dim)  # 生成每个包的实例数据
            bag_label = np.random.randint(2)  # 随机生成每个包的标签，0 或 1
            self.data.append(bag_data)
            self.labels.append(bag_label)

    def __len__(self):
        return self.num_bags

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 创建一个示例数据集
num_bags = 1000  # 假设我们有1000个包
instances_per_bag = 64  # 每个包中有64个实例
input_dim = 64  # 每个实例的维度为64

dataset = MILDataset(num_bags, instances_per_bag, input_dim)
# 创建示例模型
instance_embedding_model = InstanceEmbedding(input_dim=64, output_dim=32)

# 创建包分类模型
bag_classification_model = BagClassification(input_dim=32, hidden_dim=64, num_classes=2)

# 假设你有一个包含64条序列的包的数据集 X 和相应的标签 Y

# 训练模型
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(instance_embedding_model.parameters()) + list(bag_classification_model.parameters()), lr=0.001)

num_epochs = 20
# 创建数据加载器
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 输出示例数据的形状和标签
for epoch in range(num_epochs):
    # 对每个包进行循环
    for bag_data, bag_labels in dataloader:  # 这里假设你有一个适当的 DataLoader
        # Instance Embedding
        instance_embeddings = []
        for instance_data in bag_data:
            instance_embedding = instance_embedding_model(instance_data)
            instance_embeddings.append(instance_embedding)
        instance_embeddings = torch.stack(instance_embeddings)

        # Bag Classification
        bag_logits = bag_classification_model(instance_embeddings)

        # 计算损失
        loss = criterion(bag_logits, bag_labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training Finished!")
