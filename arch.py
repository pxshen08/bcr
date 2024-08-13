import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BCRNet']

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

# 定义神经网络模型类
class BCRNet(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(BCRNet, self).__init__()
        # nb_filter = [32, 64, 128, 256, 512]
        self.embedding_dim=embedding_dim
        self.nhead = 4
        self.heads = 2
        # Task A 输出层
        self.fc_A = nn.Linear(embedding_dim, num_classes)
        # self.fc_A = nn.Linear(embedding_dim, 1)
        # Task B 输出层
        self.fc_B = nn.Linear(embedding_dim, 1)
        # self.fc = nn.Linear(input_size, 2)
        #L=32
        # Instance embedding layer: L * 512 -> L * 32
        self.instance_embedding = nn.Linear(512, embedding_dim)

        # Per-residue attention
        self.per_residue_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=self.nhead)

        # Per-residue values
        self.per_residue_values = nn.Linear(embedding_dim, self.heads * embedding_dim)

        # Final transformation to 1 * 32
        self.final_transform = nn.Linear(self.heads * embedding_dim, embedding_dim)
        # # Classifier layer
        # self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        # Task A 输出 x.shape(32,512)
        # output_A =  torch.sigmoid(self.fc_A(x))
        # Instance embedding: L * 512 -> L * 32
        # self.L = 10
        # self.nhead = 4
        # # self.heads = 2
        x1 = self.instance_embedding(x)# torch.Size([32, 32])降维
        # x23=self.fc_A(x1)
        # x24=x23.unsqueeze(0)
        # x31 = x24.repeat(self.nhead, 1, 1)#pre residue attentiontorch.Size([4, 32, 2])
        # x2 = x1.unsqueeze(0)#(1.32.32)
        # x22=x1.unsqueeze(2)#(32.32.1)
        # x32 = x2.repeat(self.nhead, 1, 1)# valuesorch.Size([4, 32, 32])torch.Size([4, 32, 256])
        # q1, q2 = self.per_residue_attention(x32, x32, x32)#q1 valuesorch.Size
        # x4= torch.matmul(q1, x31)
        # output_A=torch.mean(x4, dim=0)
        # Per-residue attention
        # Shape: nhead * L * 32 4,32,32
        # x_att, _ = self.per_residue_attention(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))
        #x_att, _ = self.per_residue_attention(x3, x3, x3)#torch.Size([10, 32, 32])
        # Shape: L * (heads * 32)(32,64)
        #x_values = self.per_residue_values(x1)#torch.Size([32, 64])
        # x_att = x_att.unsqueeze(-1).expand(-1, -1, -1, self.heads).contiguous().view(self.L, -1, self.heads * self.embedding_dim)
        # Concatenate per-residue attention and values
        # Shape: L * (nhead * 32) 32,4*32
        # x_concat = torch.cat([x_att, x_values], dim=2)

        # Final transformation to 1 * 32
        # Shape: 1 * 32
        # x_final = self.final_transform(x_concat.mean(dim=0))
        output_A = self.fc_A(x1)
        # Task B 输出
        output_B = self.fc_B(x1)
        return output_A, output_B


class MyNetwork(nn.Module):
    def __init__(self, L, nhead, heads):
        super(MyNetwork, self).__init__()
        self.L = L
        self.nhead = nhead
        self.heads = heads

        # Instance embedding layer: L * 512 -> L * 32
        self.instance_embedding = nn.Linear(512, 32)

        # Per-residue attention
        self.per_residue_attention = nn.MultiheadAttention(embed_dim=32, num_heads=nhead)

        # Per-residue values
        self.per_residue_values = nn.Linear(32, heads * 32)

        # Final transformation to 1 * 32
        self.final_transform = nn.Linear(heads * 32, 32)

    def forward(self, x):
        # Instance embedding: L * 512 -> L * 32
        x = self.instance_embedding(x)

        # Per-residue attention
        # Shape: nhead * L * 32
        x_att, _ = self.per_residue_attention(x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2))

        # Per-residue values
        # Shape: L * (heads * 32)
        x_values = self.per_residue_values(x)

        # Concatenate per-residue attention and values
        # Shape: L * (nhead * 32)
        x_concat = torch.cat([x_att, x_values.view(self.L, self.heads * 32)], dim=2)

        # Final transformation to 1 * 32
        # Shape: 1 * 32
        x_final = self.final_transform(x_concat.mean(dim=0))

        return x_final

if __name__ == "__main__":
    # Example usage:
    L = 10  # Example sequence length
    nhead = 4  # Number of heads for multihead attention
    heads = 2  # Number of heads for per-residue values
    num_classes = 5  # Example number of classes
    embedding_dim=32
    model = BCRNet(embedding_dim ,num_classes).cuda()
    input_tensor = torch.randn(32, 512).cuda()  # Example input tensor
    output_A, output_B = model(input_tensor)
    print(output_A.shape)  # Output shape: torch.Size([10, 5])