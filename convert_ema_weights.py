from collections import OrderedDict

import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str)
    parser.add_argument("--output-path", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    print(checkpoint.keys())
    checkpoint.pop("n_averaged")
    new_checkpoint = OrderedDict(
        {k.replace("module.", ""): v for k, v in checkpoint.items()}
    )
    new_checkpoint = OrderedDict(
        {k.replace("_orig_mod.", ""): v for k, v in new_checkpoint.items()}
    )
    print(new_checkpoint.keys())
    torch.save(new_checkpoint, args.output_path)

if __name__ == "__main__":
    main()