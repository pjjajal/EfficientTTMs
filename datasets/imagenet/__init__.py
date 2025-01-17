from typing import Sequence

import json
import torch
import torchvision.transforms.v2 as tvt
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.datasets import ImageNet


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> tvt.Normalize:
    return tvt.Normalize(mean=mean, std=std)


def create_imagenet_dataset(args, image_size=224):

    if args.pretraining:
        transforms = [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(image_size),
            tvt.RandomHorizontalFlip(),
        ]
        if not (args.augmentations or args.rand_aug):
            transforms.append(make_normalize_transform())
        train_transform = tvt.Compose(transforms)
    else:
        transforms = [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(image_size),
            tvt.RandomHorizontalFlip(),
        ]
        if args.augmentations:
            print("Using 3aug.")
            transforms.extend(
                [
                    tvt.RandomChoice(
                        [
                            tvt.GaussianBlur(7),
                            tvt.RandomSolarize(threshold=0.5, p=1),
                            tvt.RandomGrayscale(p=1),
                        ]
                    ),
                    tvt.ColorJitter(0.3, 0.3, 0.3, 0.2),
                ]
            )
        elif args.rand_aug:
            print("Using RandAugment.")
            transforms.append(
                tvt.RandAugment(
                    num_ops=args.num_ops,
                    magnitude=args.magnitude,
                )
            )
        transforms.append(make_normalize_transform())
        train_transform = tvt.Compose(transforms)

    val_transform = tvt.Compose(
        [
            tvt.Resize(256, interpolation=tvt.InterpolationMode.BICUBIC),
            tvt.CenterCrop(224),
            tvt.ToTensor(),
            make_normalize_transform(),
        ]
    )

    train_dataset = ImageNet(args.data_path, "train", transform=train_transform)
    val_dataset = ImageNet(args.data_path, "val", transform=val_transform)

    return train_dataset, val_dataset
