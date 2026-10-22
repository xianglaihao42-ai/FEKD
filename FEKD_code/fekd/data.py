"""CIFAR-100 训练数据及图片推理共用的预处理。"""

from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from .utils import worker_seed

MEAN = (0.5071, 0.4867, 0.4408)
STD = (0.2675, 0.2565, 0.2761)


def image_transform(training=False):
    steps = []
    if training:
        steps += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    return transforms.Compose(steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def training_loader(data_root, batch_size, workers, seed, device, download):
    dataset = datasets.CIFAR100(
        str(data_root), train=True, transform=image_transform(training=True), download=download
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=worker_seed,
        generator=torch.Generator().manual_seed(seed),
    )
