import kornia.contrib as K
import torch.nn as nn
import torch


class ChessPiecePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = K.VisionTransformer(
            image_size=50, patch_size=16, in_channels=1, embed_dim=128, num_heads=3
        )
        self.out = K.ClassificationHead(num_classes=13)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)

        x = self.inp(x)
        x = self.out(x)

        return x
