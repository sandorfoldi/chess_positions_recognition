import os
from typing import List

import numpy as np
from PIL import Image

def crop(image: np.ndarray) -> List[np.ndarray]:
    parts = []
    for r in range(0, image.shape[0], 50):
        for c in range(0, image.shape[1], 50):
            parts.append(image[r: r + 50, c:c + 50, :])
    return parts


def transform_label(filename: str) -> List[str]:
    orig_label = filename.split(".")[0]
    transformed_label = []
    ranks = orig_label.split("-")
    for rank in ranks:
        for letter in rank:
            if letter in list("0123456789"):
                transformed_label.extend(list(int(letter) * "E"))
            else:
                transformed_label.append(letter)
    return transformed_label


def make_dataset(data_size: int = 100,
    input_dir: str = "data/raw/train", output_dir: str = "data/processed/train") -> None:
    files = os.listdir(input_dir)[:data_size]
    for idx, file in enumerate(files):
        folder_names = transform_label(file)
        for name in folder_names:
            if not name.isupper():
                os.makedirs(f"{output_dir}/b_{name}", exist_ok=True)
            else:
                os.makedirs(f"{output_dir}/w_{name}", exist_ok=True)
        orig_image = Image.open(f"{input_dir}/{file}")
        orig_image = np.array(orig_image)
        cropped_images = crop(orig_image)
        for i in range(len(folder_names)):
            image = Image.fromarray(cropped_images[i])
            if not folder_names[i].isupper():
                image.save(f"{output_dir}/b_{folder_names[i]}/{idx}-{i}.jpeg")
            else:
                image.save(f"{output_dir}/w_{folder_names[i]}/{idx}-{i}.jpeg")


if __name__ == "__main__":
    make_dataset()
