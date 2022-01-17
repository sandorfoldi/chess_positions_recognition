import torch
import hydra
from src.models.model import ChessPiecePredictor
import omegaconf


def test_model_output():
    cfg = omegaconf.OmegaConf.load('src\conf\config.yaml')
    print(cfg)
    shape_tensor = torch.randn(cfg.batch_size, 1, cfg.image_size, cfg.image_size)
    model = ChessPiecePredictor()
    assert list(model(shape_tensor).shape) == [cfg.batch_size, 13] , "Model didn't output the right shape"
