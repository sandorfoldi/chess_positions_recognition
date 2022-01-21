import hydra
import torch
import wandb
from model import ChessPiecePredictor
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from kornia.x import ImageClassifierTrainer, ModelCheckpoint
import random
from model_checkpoint import MyModelCheckpoint
import os


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):

    # This is secret and shouldn't be checked into version control
    os.environ["WANDB_API_KEY"] = "50f569476fd1824505d9afdd6374b1cafa309ce1"
    # Name and notes optional
    WANDB_ENTITY = "mdjska"
    WANDB_PROJECT = "chess-position"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    print(f"Training started with parameters: {cfg}")

    torch.manual_seed(cfg.seed)

    model = ChessPiecePredictor(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=cfg.in_channels,
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
    )

    t = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=cfg.in_channels),
            transforms.ToTensor(),
        ]
    )

    train_data = ImageFolder(f"{cfg.data_path}/train", transform=t)
    valid_data = ImageFolder(f"{cfg.data_path}/test", transform=t)

    indices_train = random.sample(range(1, 60000), 5000)
    indices_valid = random.sample(range(1, 30000), 1000)

    train_data = data_utils.Subset(train_data, indices_train)
    valid_data = data_utils.Subset(valid_data, indices_valid)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.num_epochs * len(train_loader)
    )

    model_checkpoint = MyModelCheckpoint(filepath="./outputs", monitor="top5",)

    # log model gradients
    wandb.watch(model, log_freq=100)

    trainer = ImageClassifierTrainer(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        cfg,
        callbacks={"on_checkpoint": model_checkpoint},
    )

    trainer.fit()


if __name__ == "__main__":
    train()

