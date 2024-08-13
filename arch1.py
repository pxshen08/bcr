import torch
import torch.nn as nn
import torch.nn.functional as F


class BCRNet(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(BCRNet, self).__init__()
        self.num_heads = 4
        self.embed_dim = embed_dim
        assert embed_dim % self.num_heads == 0
        self.head_dim = embed_dim // self.num_heads

        self.instance_embedding = nn.Linear(512, embed_dim)
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)

        self.fc_o = nn.Linear(embed_dim, num_classes)

        self.fc_B = nn.Linear(embed_dim, 1)

    def forward(self, x):

        x = self.instance_embedding(x)
        # Linear transformations for query, key, and value
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)

        # Reshape Q, K, and V for multi-heads (sequence length = 1)
        Q = Q.view(Q.size(0), 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(K.size(0), 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attn_scores = F.softmax(scores, dim=-1)

        # Weighted sum
        weighted_sum = torch.matmul(attn_scores, V)

        # Concatenate heads
        concat_heads = weighted_sum.transpose(1, 2).contiguous().view(Q.size(0), -1, self.embed_dim)

        # Linear transformation for output
        output_A = self.fc_o(concat_heads.mean(dim=1))  # Taking the mean over sequence dimension
        output_B = self.fc_B(x)
        return output_A,output_B


if __name__ == "__main__":

    embed_dim = 32  # Example embedding dimension
    num_heads = 4  # Example number of heads
    num_classes = 10  # Example number of classes
    model = BCRNet(embed_dim, num_heads, num_classes)
    input_tensor = torch.randn(4, embed_dim)  # Example input tensor (batch_size, embed_dim)
    output = model(input_tensor)
    print(output.shape)  # Output shape: torch.Size([4, 10])