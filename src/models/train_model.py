
import hydra
import torch
from model import ChessPiecePredictor
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from kornia.x import ImageClassifierTrainer, ModelCheckpoint
import random


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
    wandb.watch(model)
    
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

    model_checkpoint = ModelCheckpoint(
        filepath="./outputs",
        monitor="top5",
    )
'''
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
'''

    print("Training started...")
    train_losses = []
    validation_losses = []

    batch_count = len(train_loader)#int(train_size / batch_size)
    epochs = 2
    for e in range(epochs):
        train_loss = 0
        train_correct = 0

        validation_loss = 0
        validation_correct = 0

        i = 0
        for images, labels in train_loader:
            # in case we use cuda to train on gpu
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            preds = model(images)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # accuracy
            _, preds_indices = torch.max(preds, dim=1)
            train_correct += (preds_indices == labels).sum()

            i += 1
            if i % 100 == 0:
                print(
                    f"Epoch: {e+1} / {epochs}"
                    f" - progress: {i} / {batch_count}"
                    f" - loss: {loss.data.mean()}"
                )

        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            validation_loss += loss.item()

            # accuracy
            _, preds_indices = torch.max(preds, dim=1)
            validation_correct += (preds_indices == labels).sum()

        train_accuracy = float(train_correct / (len(train_loader) * batch_size))
        validation_accuracy = float(validation_correct / (len(validation_loader) * batch_size))

        wandb.log({
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
        })

        start_time = time()
        train_losses.append(train_loss / len(train_loader))
        validation_losses.append(validation_loss / len(validation_loader))

    # plotting
    plt.plot(list(range(1, len(train_losses) + 1)), train_losses, label="Training loss")
    print("Train losses:", train_losses)

    plt.plot(list(range(1, len(validation_losses) + 1)), validation_losses, label="Validation loss")
    print("Validation losses:", validation_losses)

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    fig_path = Path("reports/figures/training_run.png")
    plt.savefig(fig_path)
    print(f"Saved training loss figure to {fig_path}")

    model_path = Path("models/trained_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Saved trained model to {model_path}")
    
    '''
    trainer.fit()
    '''


if __name__ == "__main__":
    train()
