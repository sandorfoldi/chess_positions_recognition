import sys
import os
import argparse
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data_utils
import hydra
import kornia as K

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from kornia.x import ImageClassifierTrainer, ModelCheckpoint, Trainer

from model import ChessPiecePredictor, CNN


class ChessPositionsDataset(Dataset):
    def __init__(self, path_to_data: str, image_transforms, split: str) -> None:
        assert split in ['train', 'test']
        self.split = split
        self.transform = image_transforms
        self.path = path_to_data
        self.labels = torch.load(self.path+'/labels.pt')

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # the tensor in which the index can be found
        # along with the place inside that tensor
        # can be calculated by division with remainder
        mod = torch.remainder(index, 2**14)
        q = (index-mod) / 2**14
        tensor_images = torch.load(self.path + '/images_'+str(int(q))+'.pt')
       
        image = tensor_images[mod]

        image = self.transform(image)

        label = self.labels[index]
        return image, label

@hydra.main(config_path="../conf", config_name="config.yaml")
def train(config) -> None:
    print("Training started...")

    torch.manual_seed(config.seed)
    
    image_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ]
    )

    # transforms are not implemented in the dataset
    train_data = ChessPositionsDataset(f'{config.data_path}/train', image_transforms, split='train')
    valid_data = ChessPositionsDataset(f'{config.data_path}/test', image_transforms, split='test')
    indices_train = torch.arange(5000)
    indices_valid = torch.arange(1000)
    
    train_data = data_utils.Subset(train_data, indices_train)
    valid_data = data_utils.Subset(valid_data, indices_valid)

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    valid_loader = DataLoader(
        valid_data, batch_size=config.batch_size, shuffle=True, num_workers=0
    )

    # batch_item = iter(train_loader).next()

    model = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,config.num_epochs * len(train_loader)
    )

    model_checkpoint = ModelCheckpoint(filepath="./outputs", monitor="top5",)

    trainer = ImageClassifierTrainer(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        config,
        callbacks={"on_checkpoint": model_checkpoint,},
    )

    trainer.fit()


if __name__ == "__main__":
    train()
