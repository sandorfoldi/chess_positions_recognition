import kornia.contrib as K
import torch
import torch.nn as nn


class ChessPiecePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = K.VisionTransformer(image_size=50, patch_size=5, in_channels=1)
        self.out = K.ClassificationHead(num_classes=13)

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