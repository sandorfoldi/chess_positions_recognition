import argparse
import sys

import matplotlib.pyplot as plt
import torch
from model import ChessPiecePredictor, CNN
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import hydra
from torchvision import transforms
from torchvision.datasets import ImageFolder
import os
from kornia.x import ImageClassifierTrainer, ModelCheckpoint
import kornia as K


@hydra.main(config_path="../conf", config_name="config.yaml")
def train(config) -> None:
    print("Training started...")

    torch.manual_seed(config.seed)

    t = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
        ]
    )

    train_data = ImageFolder(f"{config.data_path}/train", transform=t)
    valid_data = ImageFolder(f"{config.data_path}/test", transform=t)

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

    # model = nn.Sequential(
    #     K.contrib.VisionTransformer(
    #         image_size=config.image_size, patch_size=config.patch_size, in_channels=1, embed_dim=config.embed_dim
    #     ),
    #     K.contrib.ClassificationHead(embed_size=config.embed_dim, num_classes=13),
    # )

    model = CNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config.num_epochs * len(train_loader)
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
