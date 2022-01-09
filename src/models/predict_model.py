import argparse
import sys

import torch
from model import ChessPiecePredictor
from torch.utils.data import DataLoader


def predict():
    """Evaluates the given model on the given data set,
    printing the accuracy."""
    # parsing
    print("Evaluating until hitting the ceiling")
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("load_model_from", default="")
    parser.add_argument("load_data_from", default="")
    args = parser.parse_args(sys.argv[1:])

    # model loading
    model = nn.Sequential(
        K.contrib.VisionTransformer(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=1,
            embed_dim=config.embed_dim,
        ),
        K.contrib.ClassificationHead(embed_size=config.embed_dim, num_classes=13),
    )
    state_dict = torch.load(args.load_model_from)
    model.load_state_dict(state_dict)
    model.eval()
    torch.set_grad_enabled(False)

    # data loading
    test_data = torch.load(args.load_data_from)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # prediction
    for images, labels in test_loader:
        ps = torch.exp(model(images))
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy = torch.mean(equals.type(torch.FloatTensor))

        print(f"Accuracy: {accuracy.item() * 100}%")


if __name__ == "__main__":
    predict()
