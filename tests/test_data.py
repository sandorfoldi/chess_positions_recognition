from tests import _PATH_DATA
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torch


train_data = ImageFolder(
    f"{_PATH_DATA}/processed/train", transform=transforms.ToTensor()
)
train_loader = DataLoader(train_data, batch_size=2048, shuffle=True, num_workers=0)

images, labels = next(iter(train_loader))


def test_cropped_images():
    assert list(images[0].shape) == [3, 50, 50]


def test_dimension():
    assert images.ndim == 4


def test_dataset_length():
    assert len(train_loader.dataset) == 1000 * 64


def test_class_dict():
    assert torch.all(torch.eq(labels.unique(), torch.arange(13)))
