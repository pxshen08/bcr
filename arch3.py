import torch.nn as nn
import torch.nn.functional as F
import torch

class BCRNet(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super(BCRNet, self).__init__()
        # Feature fusion network
        self.input_channels = 512
        self.output_channels = embed_dim
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(in_channels=self.input_channels, out_channels=embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )
        # Output layer
        self.fc = nn.Linear(embed_dim, num_classes)
        self.fc_B = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # Feature fusion
        x = x.unsqueeze(2)  # Add a singleton dimension for sequence_length
        # Feature fusion
        x = self.feature_fusion(x)
        x = x.squeeze()  # Remove the singleton dimension
        # Output layer
        x = self.fc(x)
        x = self.feature_fusion(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Output layer
        output_A = self.fc(x)
        print(x.shape)
        output_B = self.fc_B(x)
        return output_A,output_B


if __name__ == "__main__":
    # Example usage
    # num_classes = 2  # Number of classes
    # input_channels = 512  # Number of input channels from the feature extraction network
    # output_channels = 4  # Number of output channels in the fusion network
    # model = BCRNet(output_channels, num_classes)
    # print(model)
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