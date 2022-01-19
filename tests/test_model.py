import torch
from src.models.model import ChessPiecePredictor
import omegaconf


def test_model_output():
    cfg = omegaconf.OmegaConf.load("src\conf\config.yaml")
    shape_tensor = torch.randn(cfg.batch_size, 1, cfg.image_size, cfg.image_size)
    model = model = ChessPiecePredictor(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
    )
    assert list(model(shape_tensor).shape) == [
        cfg.batch_size,
        13,
    ], "Model didn't output the right shape"