from .imagenet import create_imagenet_dataset
from .imagenet21k import create_imagenet21k_dataset


def create_dataset(args):
    train_dataset = None
    test_dataset = None
    if args.dataset == "imagenet":
        train_dataset, test_dataset = create_imagenet_dataset(args, args.image_size if hasattr(args, "image_size") else 224) 
    elif args.dataset == "imagenet-21k":
        train_dataset, test_dataset = create_imagenet21k_dataset(args, args.image_size if hasattr(args, "image_size") else 224)
    return train_dataset, test_dataset
