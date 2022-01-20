import hydra
import torch
from model import ChessPiecePredictor
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from kornia.x import ImageClassifierTrainer, ModelCheckpoint
from google.cloud import storage


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):

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

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.num_epochs * len(train_loader)
    )

    model_checkpoint = ModelCheckpoint(
        filepath="./outputs",
        monitor="top5",
    )

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

    storage_client = storage.Client()
    bucket = storage_client.bucket("chess_predictor")
    blob = bucket.blob("model_blob")

    blob.upload_from_filename("outputs/model_0.pt")


if __name__ == "__main__":
    train()
