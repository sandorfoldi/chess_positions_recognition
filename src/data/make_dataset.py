import os
from typing import List
import glob

import numpy as np
from PIL import Image
from tqdm import tqdm
from numba import njit
from tqdm import tqdm
import torch
from torchvision import transforms

class_dict = {'e': 0, 'E': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5, 'p': 6, 'R': 7, 'N': 8, 'B': 9, 'Q': 10, 'K': 11, 'P': 12}

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
    input_dir: str = "data/raw/train", output_dir: str = "data/processed/train_1"
) -> None:

    list_img = glob.glob(os.path.join(input_dir,'*.jpeg'))[:1024]

    squares_per_slice = 2**14
    images_per_slice = int(squares_per_slice / 64)
    
    square_size = (50, 50, 3)

    num_images = len(list_img)
    num_squares = num_images * 64 
    num_slices = int(num_images / images_per_slice) 

    list_labels = []
    for image_path in list_img:
        list_labels.extend(transform_label(image_path.split('/')[-1]))
    list_labels = list(map(lambda l: class_dict[l], list_labels))
    tensor_labels = torch.Tensor(list_labels)
    torch.save(tensor_labels, 'data/processed/train_1/labels_train.pt')

    for slice_ind in range(num_slices):
        t = torch.Tensor(squares_per_slice, square_size[0],square_size[1], square_size[2], )
        for image_ind in range(images_per_slice):
            img = np.array(Image.open(list_img[image_ind]))
            squares = crop(img)
            squares = np.stack(squares, axis=0)
            t[image_ind*64:(image_ind+1)*64] = torch.from_numpy(squares)

        torch.save(t, output_dir+'/imgs_train_'+str(slice_ind)+'.pt')
        
    
if __name__ == "__main__":
    make_dataset()
