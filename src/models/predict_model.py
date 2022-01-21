import argparse
import sys
from typing import List
from model import ChessPiecePredictor

import numpy as np
import torch
import re
import wandb
import hydra
from numba import njit
from PIL import Image
from torchvision import transforms
import os, random


@njit
def crop(image: np.ndarray) -> List[np.ndarray]:
    """This method was taken from src/data/make_dataset.py but modified
    to grayscale instead of rgb images"""
    parts = []
    for r in range(0, image.shape[0], 50):
        for c in range(0, image.shape[1], 50):
            parts.append(image[r : r + 50, c : c + 50])
    return parts


def transform_label(filename: str) -> List[str]:
    jpg_name_split = re.split("[-,.]", filename)[:-1]
    board_squares = "".join(jpg_name_split)

    class_dict = {
        "e": 0,
        "E": 0,
        "b": 1,
        "k": 2,
        "n": 3,
        "p": 4,
        "q": 5,
        "r": 6,
        "B": 7,
        "K": 8,
        "N": 9,
        "P": 10,
        "Q": 11,
        "R": 12,
    }
    labels = np.zeros((64, 1))

    counter = 0
    for i, square in enumerate(board_squares):
        if square.isnumeric():
            counter += int(square)
        else:
            labels[counter] = class_dict[square]
            counter += 1

    return (labels.T).tolist()[0]


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

    return fen[:-1]  # do [:-1] to remove last "-"


# @hydra.main(config_path="../conf", config_name="config")
def predict():
    """Given a model and an image of an entire chess board (found in data/raw/ folder),
    this method predicts what piece (if any) is on each square"""

    # This is secret and shouldn't be checked into version control
    os.environ["WANDB_API_KEY"] = "50f569476fd1824505d9afdd6374b1cafa309ce1"
    # Name and notes optional
    WANDB_ENTITY = "mdjska"
    WANDB_PROJECT = "chess-position"
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    # parsing
    print("Predicting squares...")
    parser = argparse.ArgumentParser(description="Evaluation arguments")
    parser.add_argument("load_model_from")
    parser.add_argument("imagedir")
    parser.add_argument("-m", "--predict_multiple", action="store_true")
    parser.add_argument("-num", "--num_img", default=10)
    args = parser.parse_args(sys.argv[1:])

    # model loading

    # if model was saved as state_dict:
    # model = ChessPiecePredictor(
    #     image_size=cfg.image_size,
    #     patch_size=cfg.patch_size,
    #     in_channels=cfg.in_channels,
    #     embed_dim=cfg.embed_dim,
    #     num_heads=cfg.num_heads,
    # )
    # model.load_state_dict(torch.load(args.load_model_from))

    # if model was saved as model:
    model = torch.load(args.load_model_from)
    model.eval()

    # randomly picking images to predict from dir
    image_dir = args.imagedir
    image_names = [random.choice(os.listdir(image_dir))]
    # if prediction of multiple images is chosen
    if args.predict_multiple:
        image_names = random.sample(os.listdir(image_dir), int(args.num_img))

    # create a wandb.Table() with columns for the images, predicted FEN, true FEN and errors
    test_data_at = wandb.Artifact(
        "test_samples_" + str(wandb.run.id), type="predictions"
    )
    columns = ["Chesssboard image", "Prediction", "True label", "Num of errors"]
    test_table = wandb.Table(columns=columns)

    for imagefile in image_names:
        image_path = os.path.join(image_dir, imagefile)
        # data loading and preparing
        image_col = Image.open(image_path)
        image = image_col.convert("L")  # convert to grayscale

        squares = crop(np.array(image))
        squares = torch.tensor(
            np.array(squares)
        )  # convert to np.array() for speed. If not, we get a user warning
        squares = transforms.ConvertImageDtype(torch.float)(squares)

        # convert from (64, image_size, image_size), models expects the extra dimension
        if model.image_size:
            squares = transforms.Resize((model.image_size, model.image_size))(squares)
            squares = squares.view(64, 1, model.image_size, model.image_size)
        else:
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

        test_table.add_data(
            wandb.Image(image_col),
            board_to_fen(board[:-1]),
            imagefile[:-5],
            sum(
                [
                    int(i != j)
                    for i, j in zip(
                        prediction_indices.tolist(), transform_label(imagefile)
                    )
                ]
            ),
        )
    test_data_at.add(test_table, "Chess_piece_predictor")
    wandb.run.log_artifact(test_data_at)


if __name__ == "__main__":
    predict()
