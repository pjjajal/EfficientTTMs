import json
import torch
import torchvision.transforms.v2 as tvt
from torchvision.datasets import ImageFolder
from utils.cfolder import CachedImageFolder
from .augment import make_normalize_transform


def create_imagenet21k_dataset(args, image_size=224):
    train_data_path = str(args.data_path) + "/imagenet21k_train"
    val_data_path = str(args.data_path) + "/imagenet21k_val"
    if args.pretraining:
        transforms = [
            tvt.ToImage(),
            tvt.ToDtype(torch.float32, scale=True),
            tvt.RandomResizedCrop(image_size),
            tvt.RandomHorizontalFlip(),
            make_normalize_transform(),
        ]
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
            tvt.CenterCrop(image_size),
            tvt.ToTensor(),
            make_normalize_transform(),
        ]
    )

    cached_data = None
    if args.cache_path:
        with open(args.cache_path, "r") as f:
            cached_data = json.load(f)
    train_dataset = CachedImageFolder(
        root=train_data_path, transform=train_transform, cached_data=cached_data
    )
    val_dataset = ImageFolder(root=val_data_path, transform=val_transform)

    return train_dataset, val_dataset

