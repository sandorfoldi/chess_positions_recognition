import torch
#from tests import _PATH_DATA

#dataset_img = torch.load(f'{_PATH_DATA}/processed/train/imgs_train_0.pt')
dataset_img = torch.load('data/processed/train/imgs_train_0.pt')
print(len(dataset_img))

def test_cropped_images():
    assert list(dataset_img[1].size()) == [50,50,3] 

