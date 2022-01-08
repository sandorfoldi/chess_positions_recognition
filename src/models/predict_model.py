import argparse
import sys

import torch
from torchvision import transforms
from model import ChessPiecePredictor
from torch.utils.data import DataLoader
import torchvision

def predict():
    """Evaluates the given model on the given data set,
    printing the accuracy."""
    # parsing
    print("Evaluating model...")
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('load_model_from', default="")
    parser.add_argument('load_data_from', default="")
    args = parser.parse_args(sys.argv[1:])

    # model loading
    model = ChessPiecePredictor()
    state_dict = torch.load(args.load_model_from)
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # data loading
    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.ImageFolder(args.load_data_from, transform=t)
    train_loader = DataLoader(train_data, batch_size=256,
                              shuffle=False, num_workers=4)

    # prediction
    accuracy = 0
    for images, labels in train_loader:
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += equals.type(torch.FloatTensor)

    accuracy = torch.mean(accuracy)
    print(f"Accuracy: {accuracy.item() * 100}%")


if __name__ == "__main__":
    predict()