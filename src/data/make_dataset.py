import os
from typing import List
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import njit

import torch
from torchvision import transforms


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

'''
def make_dataset(
    input_dir: str = "data/raw/train", output_dir: str = "data/processed/train"
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

    files = os.listdir(input_dir)[:20000]

    # for idx, file in enumerate(tqdm(files)):
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
'''

def make_dataset(
    input_dir: str = "data/raw/train", output_dir: str = "data/processed/train_1"
) -> None:
    list_raw_img = glob.glob(os.path.join(input_dir,'*.jpeg'))
    img_per_slice = 100000
    img_size_processed = (50, 50, 3)
    slice_num = int(len(list_raw_img) * 64 / img_per_slice) # there might be some problems later with rounding integers here
    # print(slice_num)
    for slice_ind in range(slice_num):
        t = torch.Tensor(img_per_slice,  img_size_processed[2], img_size_processed[0],img_size_processed[1],)
        print(t.shape)
        for i in range(img_per_slice):
            print(t[i].shape)
            print(transforms.ToTensor()(Image.open(list_raw_img[slice_ind*img_per_slice+i])).shape)
            break
            t[i] = transforms.ToTensor()(Image.open(list_raw_img[slice_ind*img_per_slice+i]))
        t.save(output_dir+'imgs_train_'+str(slice_ind))
        break
    

if __name__ == "__main__":
    make_dataset()
