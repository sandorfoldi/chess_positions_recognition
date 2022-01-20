import argparse
import math
import sys
from pathlib import Path
from time import time
import random

import matplotlib.pyplot as plt
import torch
import torchvision
from model import ChessPiecePredictor, CNN
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import transforms
from torchvision.datasets import ImageFolder
import wandb

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
from tqdm import tqdm
import wandb


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
    train_losses, test_losses, validation_acc, training_acc = [], [], [], []

    for e in range(cfg.num_epochs):
        running_loss = 0
        running_train_accuracy = 0 
        print(f'epoch: {e}')
        i = 0
        for images, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            
            optimizer.zero_grad()

            out = model(images)

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()
            train_accuracy = torch.mean(equals.type(torch.FloatTensor))
            running_train_accuracy += train_accuracy.item()

            i += 1
            if i % (len(train_loader)//5) == 0:
                wandb.log({
                    "Training Loss":loss.data.mean().item()
                })
            
            


        else:
            with torch.no_grad():
                running_val_accuracy = 0
                running_val_loss = 0
                j = 0
                for images, labels in valid_loader:
                    out = model(images)
                    loss = criterion(out, labels)
                    running_val_loss += loss.item()
                    top_p, top_class = out.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_accuracy = torch.mean(equals.type(torch.FloatTensor))
                    running_val_accuracy += val_accuracy.item()

                    j += 1
                    if j % len(valid_loader)//5 == 0:
                        wandb.log({
                            "Validation Loss": loss.data.mean().item(),
                        })

            epoch_loss = running_loss / len(train_loader)
            epoch_train_acc = running_train_accuracy / len(train_loader)
            epoch_val_loss = running_val_loss / len(valid_loader)
            epoch_val_acc = running_val_accuracy / len(valid_loader)
            

            train_losses.append(epoch_loss)
            test_losses.append(epoch_val_loss)
            validation_acc.append(epoch_val_acc)
            training_acc.append(epoch_train_acc)

            print(f"Testset accuracy: {epoch_val_acc*100}%")
            print(f"Validation loss: {epoch_val_loss}")
            print(f"Training loss: {epoch_loss}")
            wandb.log({
                #"train_loss": train_loss,
                #"validation_loss": validation_loss,
                #"train_accuracy": train_accuracy,
                "Validation Accuracy": epoch_val_loss*100,
                "Training Accuracy": epoch_train_acc*100    
            })
    print("Training finished!")

    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/trained_model.pt")
    print("Model saved")
    
    plt.plot(np.arange(cfg.num_epochs), train_losses, label="training loss")
    plt.plot(np.arange(cfg.num_epochs), test_losses, label="validation loss")
    plt.plot(np.arange(cfg.num_epochs), validation_acc, label="validation accuracy")
    plt.plot(np.arange(cfg.num_epochs), training_acc, label="training accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.title("model training")

    os.makedirs("figures/", exist_ok=True)
    plt.savefig("figures/train_loss.png")

if __name__ == "__main__":
    train()


#def train():
    '''
    #parser = argparse.ArgumentParser(description="Training arguments")
    #parser.add_argument("load_data_from", default="")
    # used for loading and training a model
    #parser.add_argument("--continue_training_from", required=False, type=Path)
    #args = parser.parse_args(sys.argv[1:])

    #start_time = time()
    wandb.init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading data...")
    t = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Grayscale(),
        ]
    )
    '''
    #data_set = torchvision.datasets.ImageFolder(args.load_data_from, transform=t)
    #train_size = math.ceil(len(data_set) * 0.85)
    #validation_size = math.floor(len(data_set) * 0.15)

    #train_data, validation_data = torch.utils.data.random_split(
    #    data_set, [train_size, validation_size]
    #)
    '''

    train_data = ImageFolder("data/processed/train", transform=t)
    validation_data = ImageFolder("data/processed/test", transform=t)

    indices_train = random.sample(range(1, 60000), 5000)
    indices_valid = random.sample(range(1, 30000), 1000)

    train_data = data_utils.Subset(train_data, indices_train)
    validation_data = data_utils.Subset(validation_data, indices_valid)

    batch_size = 4
    train_loader = DataLoader(
        train_data, batch_size=batch_size, pin_memory=True, num_workers=4, shuffle=True
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=4,
        shuffle=False,
    )

    #model = ChessPiecePredictor()
    model = CNN()
    # TODO what does this do?
    wandb.watch(model)

    if args.continue_training_from:
        print(f"Loading model from {args.continue_training_from} ...")
        checkpoint = torch.load(args.continue_training_from)
        model.load_state_dict(checkpoint)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    learning_rate = 1e-4
    momentum = 0.9
    weight_decay = 0.0000001
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    print("Training started...")
    train_losses = []
    validation_losses = []

    train_batch_count = len(train_loader)#int(train_size / batch_size)
    epochs = 2
    for e in range(epochs):
        running_train_loss = 0
        train_correct = 0

        running_validation_loss = 0
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
            running_train_loss += loss.item()

            # accuracy
            _, preds_indices = torch.max(preds, dim=1)
            train_correct += (preds_indices == labels).sum()

            i += 1
            if i % (len(train_loader)//5) == 0:
                print(
                    f"Epoch: {e+1} / {epochs}"
                    f" - progress: {i} / {len(train_loader)}"
                    f" - loss: {loss.data.mean()}"
                )
                wandb.log({
                    "Training Loss":loss.data.mean().item()
                })

        j=0
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)
            loss = criterion(preds, labels)

            running_validation_loss+= loss.item()

            # accuracy
            _, preds_indices = torch.max(preds, dim=1)
            validation_correct += (preds_indices == labels).sum()
            
            j += 1
            if j % len(validation_loader)//5 == 0:
                wandb.log({
                    "Validation Loss": loss.data.mean().item(),
                        })
                print(
                    f"Epoch: {e+1} / {epochs}"
                    f" - progress: {j} / {len(validation_loader)}"
                    f" - loss: {loss.data.mean()}"
                )


        train_accuracy = float(train_correct / (len(train_loader) * batch_size))
        validation_accuracy = float(validation_correct / (len(validation_loader) * batch_size))

        wandb.log({
            #"epoch": e + 1,
            #"train_loss": train_loss,
            #"validation_loss": validation_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            #"time": time,
        })
        
        print("Epoch:", e + 1)
        print("Train loss:         ", running_train_loss / len(train_loader))
        print("Validation loss:    ", running_validation_loss/ len(validation_loader))
        print("Train Accuracy:     ", train_accuracy)
        print("Validation accuracy:", validation_accuracy)
        print("Time:               ", time() - start_time)
        

        start_time = time()
        train_losses.append(running_train_loss / len(train_loader))
        validation_losses.append(running_validation_loss / len(validation_loader))

    # plotting
    plt.plot(list(range(1, len(train_losses) + 1)), train_losses, label="Training loss")
    print("Train losses:", train_losses)

    #wandb.log({"train_losses": train_losses})

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



'''
if __name__ == "__main__":
    train()
'''