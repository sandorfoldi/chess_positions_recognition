import argparse
import sys
from typing import List

import numpy as np
import torch
from model import ChessPiecePredictor
from PIL import Image
from torchvision import transforms


def crop(image: np.ndarray) -> List[np.ndarray]:
    """This method was taken from src/data/make_dataset.py but modified
    to grayscale instead of rgb images"""
    parts = []
    for r in range(0, image.shape[0], 50):
        for c in range(0, image.shape[1], 50):
            parts.append(image[r: r + 50, c: c + 50])
    return parts


def board_to_fen(board):
    board_len = 8

    lines = board.split("\n")

    fen = ""
    for line in lines:
        for i in range(board_len):
            if line[i] == "_":
                if len(fen) == 0:
                    fen += "1"
                elif fen[-1] in "0123456789":
                    fen = fen[:-1] + str(int(fen[-1]) + 1)
                else:
                    fen += "1"
            else:
                fen += line[i]

        fen += "-"

    return fen[:-1]  # do [:-1] to remove last "/"


def predict():
    """Given a model and an image of an entire chess board (found in data/raw/ folder),
    this method predicts what piece (if any) is on each square"""
    # parsing
    print("Predicting squares...")
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("load_model_from", default="")
    parser.add_argument("image", default="")
    args = parser.parse_args(sys.argv[1:])

    # model loading
    model = ChessPiecePredictor()
    state_dict = torch.load(args.load_model_from)
    model.load_state_dict(state_dict)
    model.eval()

    # data loading and preparing
    image = Image.open(args.image)
    image = image.convert("L")  # convert to grayscale

    squares = crop(np.array(image))
    squares = torch.tensor(
        np.array(squares)
    )  # convert to np.array() for speed. If not, we get a user warning
    squares = transforms.ConvertImageDtype(torch.float)(squares)

    # convert from (64, 50, 50), models expects the extra dimension
    squares = squares.view(64, 1, 50, 50)

    with torch.no_grad():
        prediction = model(squares)

    # show results
    classes = ["_", "b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]

    _, prediction_indices = torch.max(prediction, dim=1)

    board = ""
    i = 0
    for index in prediction_indices:
        print(classes[index], end="")
        board += str(classes[index])
        i += 1
        if i % 8 == 0:
            print()
            board += "\n"

    # the last char is a newline, so [:-1] removes that
    print("fen:", board_to_fen(board[:-1]))


if __name__ == "__main__":
    predict()
