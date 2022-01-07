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
    '''
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description='Evaluation arguments')
    parser.add_argument('load_model_from', default="")
    parser.add_argument('load_data_from', default="")
    args = parser.parse_args(sys.argv[1:])
    '''

    # model loading
    model = ChessPiecePredictor()
    #state_dict = torch.load(args.load_model_from)
    state_dict = torch.load("../../models/trained_model.pth")
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # data loading
    #test_data = torch.load(args.load_data_from)
    #test_loader = DataLoader(test_data, batch_size=len(test_data),
    #                         shuffle=False)

    t = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.ImageFolder("../../data/processed/train", transform=t)
    train_loader = DataLoader(train_data, batch_size=256,
                              shuffle=False, num_workers=4)

    i = 0
    # prediction
    for images, labels in train_loader:
        i += 1
        if i <= 10:
            continue
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        print(f'Accuracy: {accuracy.item() * 100}%')


if __name__ == "__main__":
    predict()