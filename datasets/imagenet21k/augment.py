from typing import Sequence

import torch
import torchvision.transforms.v2 as tvt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


class Augmentation:
    def __init__(self) -> None:
        self.transform = tvt.Compose(
            [
                tvt.RandomChoice(
                    [
                        tvt.GaussianBlur(7),
                        tvt.RandomSolarize(threshold=0.5, p=1),
                        tvt.RandomGrayscale(p=1),
                    ]
                ),
                make_normalize_transform(),
            ]
        )

    def __call__(self, x):
        return self.transform(x)