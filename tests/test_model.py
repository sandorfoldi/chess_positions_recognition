import torch
from src.models.model import ChessPiecePredictor


shape_tensor = torch.randn(50,50,3)
print(shape_tensor.shape)
model = ChessPiecePredictor()
print(model(shape_tensor).shape)

def test_model_output():
    shape_tensor = torch.randn(2500)
    model = ChessPiecePredictor()
    assert list(model(shape_tensor).shape) == [64, 10] , "Model didn't output the right shape"
