'''
import torch
from torch import Tensor
from src.models.train_model import train
from src.models.model import ChessPiecePredictor
from unittest import mock

#from tests import _PATH_DATA

#from src.data.make_dataset import


#dataset_img = torch.load(f'{_PATH_DATA}/processed/train/imgs_train_0.pt')
#dataset_label = torch.load(f'{_PATH_DATA}/processed/train/labels_train.pt')
#dataset_label = torch.load('data/processed/train/labels_train.pt')

#print(dataset_label.size(dim=0))

@mock.patch('src.models.train_model.ChessPiecePredictor')
def test_train(mock_ChessPiecePredictor):
    try:
        train()
    except FileNotFoundError:
        pass


    
    
    except ValueError:
    #    print('Ignore parsing')
    #    print(mock_ChessPiecePredictor)
    #    print('ignore optimzer')
    #except ValueError:
        pass
    #resp = import.train_model.train()
    #train()

    mock_ChessPiecePredictor.assert_called_once()
'''
    