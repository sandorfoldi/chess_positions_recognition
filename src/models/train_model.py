import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CNN, ChessPiecePredictor
from torch import nn, optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from kornia.x import ImageClassifierTrainer, ModelCheckpoint, Trainer
import os
import random


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):

    print(f"Training started with parameters: {cfg}")
    
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(cfg.seed)

    model = CNN()

    t = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    train_data = ImageFolder(f"{cfg.data_path}/train", transform=t)
    valid_data = ImageFolder(f"{cfg.data_path}/test", transform=t)

    indices_train = random.sample(range(1, 60000), 5000)
    indices_valid = random.sample(range(1, 30000), 1000)

    train_data = data_utils.Subset(train_data, indices_train)
    valid_data = data_utils.Subset(valid_data, indices_valid)
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=cfg.batch_size, shuffle=True
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    step = 0
    train_losses, test_losses, accuracies = [], [], []
    
    for e in range(cfg.num_epochs):
        running_loss = 0
        for images, labels in train_loader:
            # print(labels)
            images = images.to(DEVICE)
            
            optimizer.zero_grad()

            out = model(images)

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            step += 1

        else:
            with torch.no_grad():
                running_accuracy = 0
                running_val_loss = 0
                for images, labels in valid_loader:
                    out = model(images)
                    loss = criterion(out, labels)
                    running_val_loss += loss.item()
                    top_p, top_class = out.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                    running_accuracy += accuracy.item()
            epoch_loss = running_loss / len(train_loader)
            epoch_val_loss = running_val_loss / len(valid_loader)
            epoch_val_acc = running_accuracy / len(valid_loader)

            train_losses.append(epoch_loss)
            test_losses.append(epoch_val_loss)
            accuracies.append(epoch_val_acc)

            print(f"Testset accuracy: {epoch_val_acc*100}%")
            print(f"Validation loss: {epoch_val_loss}")
            print(f"Training loss: {epoch_loss}")
    print("Training finished!")
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, cfg.num_epochs * len(train_loader)
    # )
    
    # model_checkpoint = ModelCheckpoint(filepath="./outputs", monitor="top5",)
    
    
    # trainer = ImageClassifierTrainer(
    #     model,
    #     train_dataloader = train_loader,
    #     valid_dataloader = valid_loader,
    #     criterion,
    #     optimizer,
    #     scheduler,
    #     cfg,
    #     callbacks={"on_checkpoint": model_checkpoint,},
    # )
    
    # trainer.fit()
    
    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Model saved")
    
    plt.plot(np.arange(cfg.num_epochs), train_losses, label="training loss")
    plt.plot(np.arange(cfg.num_epochs), test_losses, label="validation loss")
    plt.plot(np.arange(cfg.num_epochs), accuracies, label="accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.title("model training")

    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/train_loss.png")


if __name__ == "__main__":
    train()
