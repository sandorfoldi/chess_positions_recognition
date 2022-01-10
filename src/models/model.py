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
