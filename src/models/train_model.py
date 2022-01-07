import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms
from model import ChessPiecePredictor
from torch import nn, optim
from torch.utils.data import DataLoader


def train():
    print("Training started...")
    #parser = argparse.ArgumentParser(description='Training arguments')
    #parser.add_argument('load_data_from', default="")
    #args = parser.parse_args(sys.argv[1:])

    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    #train_data = torch.load(args.load_data_from)
    train_data = torchvision.datasets.ImageFolder("../../data/processed/train", transform=t)
    train_loader = DataLoader(train_data, batch_size=256,
                              shuffle=False, num_workers=4)

    model = ChessPiecePredictor()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    losses = []

    epochs = 1
    i = 0
    for e in range(epochs):
        print("epoch:", e)
        running_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print(i)
            i+=1

        print("loss:", running_loss/len(train_loader))
        losses.append(running_loss)

    # plotting
    plt.plot(list(range(epochs)), losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig("../../reports/figures/training_run.png")
    torch.save(model.state_dict(), '../../models/trained_model.pth')


if __name__ == '__main__':
    train()
