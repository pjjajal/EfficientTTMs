import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.benchmark as bench
from fvcore.nn.activation_count import ActivationCountAnalysis
from fvcore.nn.flop_count import FlopCountAnalysis
from torch.utils.data import DataLoader

from models.deit.factory import deit_factory
from models.vit.factory import vit_factory
from models.vittm.factory import deittm_factory, vittm_factory
from models.tome import apply_patch
from models.lookup.factory import lookupvit_factory
import timm

BENCHMARKS_PATH = Path("benchmarks")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Models")

    # General Arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=["mps", "cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--measure-flops", default=False, action="store_true")

    # Model Specifcation
    parser.add_argument(
        "--model",
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
            "vit_base32",
            "vit_large",
            "deit_tiny",
            "deit_small",
            "deit_base",
            "deit_large",
            "vittm_tiny",
            "vittm_small",
            "vittm_base",
            "vittm_large",
            "deittm_tiny",
            "deittm_small",
            "deittm_base",
            "deittm_large",
            "lookupvit_3x3",
            "lookupvit_5x5",
            "lookupvit_7x7",
            "lookupvit_10x10",
        ],
        default="vit_tiny",
    )
    # Model Specific Arguments
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--memory-ps", type=int, default=16)
    parser.add_argument("--process-ps", type=int, default=32)
    parser.add_argument("--process-tokens", type=int, default=None)
    parser.add_argument(
        "--process-embedded-type",
        choices=["patch", "downsample", "latent"],
        default="patch",
    )
    parser.add_argument(
        "--rw-head-type", choices=["tl", "ca", "la", "lca", "lin", "dyna"], default="tl"
    )
    parser.add_argument(
        "--fusion-type", choices=["residual", "erase", "add_erase"], default="residual"
    )

    # rw specific
    parser.add_argument("--latent-size-scale", type=int, default=1)
    parser.add_argument("--reduced-dim", type=int, default=2)
    parser.add_argument("--dyna-num-heads", type=int, default=16)
    parser.add_argument("--dyna-concat", action="store_true")

    # memory blocks specific
    parser.add_argument(
        "--memory-blocks", choices=["none", "mlp", "conv"], default="none"
    )
    parser.add_argument("--memory-mlp-ratio", type=float, default=0.5)

    parser.add_argument("--memory-encoder-depths", nargs="+", type=int, default=[3, 3])
    parser.add_argument(
        "--memory-encoder-downsamples", nargs="+", type=int, default=[2, 1]
    )
    parser.add_argument("--memory-decoder-depths", nargs="+", type=int, default=[3, 3])
    parser.add_argument(
        "--memory-decoder-upsamples", nargs="+", type=int, default=[2, 1]
    )

    # ToMe
    parser.add_argument("--tome", action="store_true")
    parser.add_argument("--tome-k", type=int, default=2)

    # Dataloader Specification
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)

    # Save Data
    parser.add_argument("--save-data", action="store_true")

    return parser.parse_args()


def create_model(args, pretrained=False):
    model_base, model_size = args.model.split("_")

    args.memory_blocks = args.memory_blocks if args.memory_blocks != "none" else ""

    if model_base == "vit":
        model, config = vit_factory(
            model_size, pretrained=pretrained, patch_size=args.patch_size
        )
        if args.tome:
            apply_patch(model)
            model.r = args.tome_k
    elif model_base == "deit":
        model = deit_factory(model_size, pretrained=pretrained)
        if args.tome:
            apply_patch(model)
            model.r = args.tome_k
    if model_base == "vittm":
        model = vittm_factory(
            model_size,
            pretrained=pretrained,
            memory_ps=args.memory_ps,
            process_ps=args.process_ps,
            latent_size_scale=args.latent_size_scale,
            rw_head_type=args.rw_head_type,
            fusion_type=args.fusion_type,
            reduced_dim=args.reduced_dim,
            dyna_num_heads=args.dyna_num_heads,
            dyna_concat=args.dyna_concat,
            memory_blocks=args.memory_blocks,
            memory_mlp_ratio=args.memory_mlp_ratio,
            memory_encoder_depths=args.memory_encoder_depths,
            memory_encoder_downsamples=args.memory_encoder_downsamples,
            memory_decoder_depths=args.memory_decoder_depths,
            process_tokens=args.process_tokens,
            process_embedded_type=args.process_embedded_type,
        )
    elif model_base == "deittm":
        model = deittm_factory(
            model_size,
            pretrained=pretrained,
            memory_ps=args.memory_ps,
            process_ps=args.process_ps,
            latent_size_scale=args.latent_size_scale,
            rw_head_type=args.rw_head_type,
            fusion_type=args.fusion_type,
            reduced_dim=args.reduced_dim,
            dyna_num_heads=args.dyna_num_heads,
            dyna_concat=args.dyna_concat,
            memory_blocks=args.memory_blocks,
            memory_mlp_ratio=args.memory_mlp_ratio,
            memory_encoder_depths=args.memory_encoder_depths,
            memory_encoder_downsamples=args.memory_encoder_downsamples,
            memory_decoder_depths=args.memory_decoder_depths,
            process_tokens=args.process_tokens,
            process_embedded_type=args.process_embedded_type,
        )
    elif model_base == "lookupvit":
        model = lookupvit_factory(name=args.model)
    return model


def device_info(args):
    device = torch.device(args.device)
    device_name = ""
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
    return device_name


### Create a benchmark function (very simple)
def benchmark_compvit_milliseconds(x: torch.Tensor, model: torch.nn.Module):
    ### Do the benchmark!
    t0 = bench.Timer(
        stmt=f"model.forward(x)",
        globals={"x": x, "model": model},
        num_threads=1,
    )

    return t0.blocked_autorange(min_run_time=8.0)


def inference(model, device, batch_size):
    ### Turn off gradient compute
    with torch.no_grad():
        ### Run Benchmark for latency, then do torch profiling!
        rand_x = torch.randn(
            size=(batch_size, 3, 224, 224), dtype=torch.float32, device=device
        )

        ### Record latency with benchmark utility
        latency_measurement = benchmark_compvit_milliseconds(rand_x, model)
        latency_mean = latency_measurement.mean * 1e3
        latency_median = latency_measurement.median * 1e3
        latency_iqr = latency_measurement.iqr * 1e3

    return latency_mean, latency_median, latency_iqr


# taken from timm: https://github.com/huggingface/pytorch-image-models/blob/d4ef0b4d589c9b0cb1d6240ff373c5508dbb8023/benchmark.py#L206
def flops(model, device, batch_size):
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((1, 3, 224, 224), device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    return fca.total(), aca.total()


def save_data(data, args):
    filename = (
        "_".join(
            [
                device_info(args).replace(" ", ""),
                args.model,
                f"bs{args.batch_size}",
            ]
        )
        + ".csv"
    )

    pd.DataFrame(data).to_csv(BENCHMARKS_PATH / filename)


def main(args):
    # Create model
    model = create_model(args)
    
    # model = timm.create_model("crossvit_15_240")
    model = model.to(args.device)
    model.eval()

    all_data = []

    if args.measure_flops:
        macs, acts = flops(model, args.device, args.batch_size)
    latency_mean, latency_median, latency_iqr = inference(
        model,
        args.device,
        args.batch_size,
    )

    print("Device: ", device_info(args))
    print("Model: ", args.model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Patch Size: ", args.patch_size)
    print("Memory Size: ", args.memory_ps)
    print("Process Size: ", args.process_ps)
    print("RW Head Type: ", args.rw_head_type)
    print("Fusion Type: ", args.fusion_type)
    print("Memory Blocks: ", args.memory_blocks)
    print("Memory MLP Ratio: ", args.memory_mlp_ratio)
    print("Latency Mean: ", latency_mean)
    print("Latency Median: ", latency_median)
    print("Latency IQR: ", latency_iqr)
    print(f"FLOPS: {macs:,} || {macs / 1e9:.2f}G") if args.measure_flops else None
    print(f"ACTS: {acts:,} || {acts /1e6:.2f}M") if args.measure_flops else None

    if args.save_data:
        all_data.append(
            {
                "device": device_info(args),
                "model": args.model,
                "patch_size": args.patch_size,
                "latency_mean": latency_mean,
                "latency_median": latency_median,
                "latency_iqr": latency_iqr,
                "macs": f"{macs / 1e9:.2f}",
                "acts": f"{acts / 1e6:.2f}",
            }
        )

        save_data(all_data, args)


if __name__ == "__main__":
    args = parse_args()

    BENCHMARKS_PATH.mkdir(exist_ok=True, parents=True)

    # pre-eval stuff
    torch.manual_seed(args.seed)
    main(args)
