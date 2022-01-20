import random

import torchdrift
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader


def load_data():
    t = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    train_data = ImageFolder("data/processed/train", transform=t)
    valid_data = ImageFolder("data/processed/test", transform=t)

    indices_train = random.sample(range(1, 60000), 5000)
    indices_valid = random.sample(range(1, 30000), 1000)

    train_data = data_utils.Subset(train_data, indices_train)
    valid_data = data_utils.Subset(valid_data, indices_valid)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True)

    return train_loader, valid_loader


def main():
    train_loader, valid_loader = load_data()

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    model = torch.load("./outputs/2022-01-19/11-45-12/outputs/model_2.pt")

    model.eval()

    drift_detection_model = torch.nn.Sequential(
        model,
        drift_detector
    )

    predictions = model(data)

    print(predictions)


if __name__ == "__main__":
    main()
