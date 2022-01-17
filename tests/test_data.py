import torch
import hydra
from tests import _PATH_DATA
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
import numpy as np


train_data = ImageFolder(f'{_PATH_DATA}/processed/train', 
        transform=transforms.ToTensor()
    )
train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
print(type(train_data))
images, labels = next(iter(train_loader))
print(type(images))
print(images.ndim)
print((images.shape))

def test_cropped_images():
    assert list(images[0].shape) == [3,50,50] 

def test_dimension():
    assert images.ndim == 4

def test_dataset_length():
    assert len(train_loader.dataset) == 1000*64

def test_class_dict():
    labels_list = []
    for images, label in (train_loader):
        labels_list.append(label)
    assert list(np.unique(np.concatenate(labels_list, axis = 0))) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]



#dataset_img = torch.load(f'{_PATH_DATA}/processed/train')
#print()

#print(type(dataset_img))
'''

@hydra.main(config_path="../src/conf", config_name="config")
def open_data(cfg):
    t = transforms.Compose(
        [
            #transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    train_data = ImageFolder(f"{cfg.data_path}/train", transform=t)
    valid_data = ImageFolder(f"{cfg.data_path}/test", transform=t)

    indices_train = torch.arange(5000)
    indices_valid = torch.arange(1000)
    #train_data = data_utils.Subset(train_data, indices_train)
    #valid_data = data_utils.Subset(valid_data, indices_valid)
    train_loader = DataLoader(
        train_data, batch_size=64, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_data, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )



    images, labels = next(iter(train_loader))
    print(type(images))
    print(images.ndim)
    print((images[0].shape))
    print(type(train_data))
    print(len(train_data))
    print(len(train_loader.dataset))
    print(list[torch.unique(labels)])

    labels_list = []
    for images, label in (train_loader):
        labels_list.append(label)

    #print(type(labels_list))
    print(np.unique(np.concatenate(labels_list, axis = 0)))

    #labels_tensor = torch.Tensor(np.asarray(labels_list))

    #print(list[torch.unique(labels_tensor)])
    

    #return print(images.shape)

if __name__ == "__main__":
    open_data()
'''









