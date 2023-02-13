import os
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CelebA
import torch.utils.data as data
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as tvu
from tqdm import tqdm
from PIL import Image
import sys
import torchvision.transforms.functional as F


def get_dataloader_cifar10(image_size, batch_size, num_data, seed, normalize=False):
    print(f"num_data:{num_data}, seed:{seed}")
    np.random.seed(seed)
    tran_transform = test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    
    if normalize == True:
        tran_transform = test_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ]
    )
    
    train_dataset = CIFAR10(
        os.path.join("~/mia_diffusion/dataset/cifar10", "cifar10_train"),
        train=True,
        download=True,
        transform=tran_transform,
    )
    test_dataset = CIFAR10(
        os.path.join("~/mia_diffusion/dataset/cifar10", "cifar10_test"),
        train=False,
        download=True,
        transform=test_transform,
    )

    dataset = data.ConcatDataset([train_dataset, test_dataset])

    indices = np.arange(50000)
    np.random.shuffle(indices)

    training_indices = indices[0:num_data]
    testing_indices = indices[num_data:num_data*2]

    training_dataset = Subset(dataset, training_indices)
    testing_dataset = Subset(dataset, testing_indices)

    train_loader = data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = data.DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    print(f"train:{len(train_loader.dataset)}")
    print(f"test:{len(test_loader.dataset)}")

    return train_loader, test_loader, [num_data, seed]


class Crop(object):
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return F.crop(img, self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2
        )


def get_dataloader_celeba(image_size, batch_size, num_data, seed, normalize=False):
    print(f"num_data:{num_data}, seed:{seed}")
    np.random.seed(seed)

    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64

    tran_transform = transforms.Compose(
        [
            Crop(x1, x2, y1, y2),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    if normalize == True:
        tran_transform = transforms.Compose(
        [
            Crop(x1, x2, y1, y2),
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ]
    )
    
    dataset = CelebA(
        root=os.path.join("~/mia_diffusion/dataset/celeba"),
        split="train",
        transform=tran_transform,
        download=False,
    )

    indices = np.arange(50000)
    np.random.shuffle(indices)

    training_indices = indices[0:num_data]
    testing_indices = indices[num_data:num_data*2]

    training_dataset = Subset(dataset, training_indices)
    testing_dataset = Subset(dataset, testing_indices)

    train_loader = data.DataLoader(
        training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    test_loader = data.DataLoader(
        testing_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )

    print(f"train:{len(train_loader.dataset)}")
    print(f"test:{len(test_loader.dataset)}")

    return train_loader, test_loader, [num_data, seed]

if __name__ == "__main__":
    train_loader, test_loader, _ = get_dataloader_celeba(image_size=64, batch_size=128, num_data=64, seed=5)
    
    tmp = train_loader.__iter__()
    img, _ = tmp.next()
    plt.axis("off")
    plt.title("Generated Images")
    print(f"img:{img.shape}")
    result = np.transpose(tvu.make_grid(img[:16], nrow=4, padding=1, normalize=False), (1,2,0))
    plt.imsave("sample_torch.png", result.cpu().numpy())
    