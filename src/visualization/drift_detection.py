import random

import torchdrift
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as data_utils
from torch.utils.data import DataLoader
from src.models.model import ChessPiecePredictor
import hydra


def load_data(data_path):
    t = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )

    train_data = ImageFolder(f"{data_path}/train", transform=t)
    valid_data = ImageFolder(f"{data_path}/test", transform=t)

    indices_train = random.sample(range(1, 60000), 5000)
    indices_valid = random.sample(range(1, 30000), 1000)

    train_data = data_utils.Subset(train_data, indices_train)
    valid_data = data_utils.Subset(valid_data, indices_valid)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=8, shuffle=True)

    return train_loader, valid_loader


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):
    train_loader, valid_loader = load_data(cfg.data_path)

    drift_detector = torchdrift.detectors.KernelMMDDriftDetector()

    model = ChessPiecePredictor(cfg.image_size, cfg.patch_size, cfg.in_channels,
                                cfg.embed_dim, cfg.num_heads)
    model.load_state_dict(torch.load("../../../models/trained_model.pth"))
    model.eval()

    drift_detection_model = torch.nn.Sequential(
        model,
        drift_detector
    )

    images, labels = next(iter(train_loader))
    p
    predictions = model(images[0][0])

    print(predictions)


if __name__ == "__main__":
    main()
