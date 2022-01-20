import kornia.contrib as K
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessPiecePredictor(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_heads):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.inp = K.VisionTransformer(
            image_size=self.image_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
        )
        self.out = K.ClassificationHead(embed_size=self.embed_dim, num_classes=13)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        x = self.out(x)

        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # out channels x h x w (when flattening for linear)
        self.linear = nn.Linear(32 * 12 * 12, 13)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.shape[0], -1)
        x = F.log_softmax(self.linear(x), dim=1)

        return x

