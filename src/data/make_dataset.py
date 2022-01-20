import os
import numpy as np

from typing import List
from PIL import Image
from tqdm import tqdm
from numba import njit

class_dict = {
    "e": 0,
    "E": 0,
    "r": 1,
    "n": 2,
    "b": 3,
    "q": 4,
    "k": 5,
    "p": 6,
    "R": 7,
    "N": 8,
    "B": 9,
    "Q": 10,
    "K": 11,
    "P": 12,
}


@njit
def crop(image: np.ndarray) -> List[np.ndarray]:
    parts = []
    for r in range(0, image.shape[0], 50):
        for c in range(0, image.shape[1], 50):
            parts.append(image[r : r + 50, c : c + 50, :])
    return parts


@njit
def transform_label(filename: str) -> List[str]:
    orig_label = filename.split(".")[0]
    transformed_label = []
    ranks = orig_label.split("-")
    for rank in ranks:
        for letter in rank:
            if letter in "0123456789":
                transformed_label.extend(
                    (ord(letter) - 48) * "E"
                )  # use ord cuz numba don't like castings
            else:
                transformed_label.append(letter)
    return transformed_label


def make_dataset(
    input_dir: str = "data/raw/train",
    output_dir: str = "data/processed/train",
    ind_start: int = 0,
    ind_stop: int = 5000,
) -> None:

    dirs = [
        "b_b",
        "b_k",
        "b_n",
        "b_p",
        "b_q",
        "b_r",
        "w_B",
        "w_K",
        "w_N",
        "w_P",
        "w_Q",
        "w_R",
        "w_E",
    ]

    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for item in dirs:
        os.makedirs(f"{output_dir}/{item}", exist_ok=True)

    files = os.listdir(input_dir)[ind_start:ind_stop]

    for idx, file in enumerate(tqdm(files)):
        folder_names = transform_label(file)
        orig_image = Image.open(f"{input_dir}/{file}")
        orig_image = np.array(orig_image)
        cropped_images = crop(orig_image)
        for i in range(len(folder_names)):
            image = Image.fromarray(cropped_images[i])
            if not folder_names[i].isupper():
                image.save(f"{output_dir}/b_{folder_names[i]}/{idx}-{i}.jpeg")
            else:
                image.save(f"{output_dir}/w_{folder_names[i]}/{idx}-{i}.jpeg")
    os.rename(f"{output_dir}/w_E", f"{output_dir}/E")


if __name__ == "__main__":
    make_dataset(
        input_dir="data/raw/train",
        output_dir="data23/processed/train",
        ind_start=0,
        ind_stop=50,
    )

    make_dataset(
        input_dir="data/raw/test",
        output_dir="data23/processed/test",
        ind_start=0,
        ind_stop=10,
    )
