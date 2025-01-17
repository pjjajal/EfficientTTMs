import argparse
import json
from datetime import datetime
from pathlib import Path
import uuid
import random

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as tvt
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import grad_norm
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy

from datasets import create_dataset
from datasets.imagenet21k.augment import Augmentation
from models.flexivit.apply import apply_flexi_patch_embed
from models.vittm.factory import deittm_factory, vittm_factory
from models.deit.factory import deit_factory
from models.vit.factory import vit_factory
from utils.schedulers import (
    CosineAnnealingWithWarmup,
    create_cosine_with_warmup,
    create_poly_with_warmup,
)

# from distributed_shampoo.distributed_shampoo import DistributedShampoo
# from distributed_shampoo.shampoo_types import AdamGraftingConfig
from timm.loss import BinaryCrossEntropy, SoftTargetCrossEntropy


DEFAULT_CHECKPOINTS_PATH = Path("./checkpoints")

torch.set_float32_matmul_precision("medium")


def parse_args():
    parser = argparse.ArgumentParser("training and evaluation script")

    # Reproducibility Arguments.
    parser.add_argument("--seed", type=int, default=245)

    # Trainer Specific Arguments.
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument(
        "--strategy", type=str, default="ddp_find_unused_parameters_true"
    )
    parser.add_argument(
        "--precision",
        choices=[
            "32-true",
            "32",
            "16-mixed",
            "bf16-mixed",
            "transformer-engine",
            "16-true",
            "bf16-true",
            "64-true",
        ],
        default="bf16-mixed",
    )
    parser.add_argument(
        "--overfit-batches",
        type=float,
        default=0,
        help="Overfit on a subset of the data for debugging purposes",
    )

    # Dataset Arguments.
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["imagenet", "imagenet-21k"],
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument(
        "--cache_save_path",
        type=Path,
        default=None,
        help="Path where to cache data",
    )

    # Training Arguments.
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--scheduler", choices=["cosine", "poly"], default="cosine")

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=64)
    parser.add_argument("--accumulations", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--optimizer", choices=["adamw", "shampoo"], default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--precondition-frequency", type=int, default=100)
    parser.add_argument("--max-preconditioner-dim", type=int, default=8192)
    parser.add_argument("--start-preconditioning-step", type=int, default=300)
    parser.add_argument("--warmup-lr", type=float, default=1e-6)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--ema", default=0.999)

    #  Model arguments.
    parser.add_argument(
        "--model",
        choices=[
            "vit_tiny",
            "vit_small",
            "vit_base",
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
        ],
        default="vittm_tiny",
    )
    parser.add_argument("--compile", default=False, action="store_true")
    parser.add_argument("--model-checkpoint", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--replace-head", default=False, action="store_true")
    parser.add_argument(
        "--patch-size", type=int, default=16
    )  # used for ViT/DeiT models
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
    parser.add_argument("--flexi-process", default=False, action="store_true")
    parser.add_argument("--drop-path-rate", type=float, default=0.0)
    parser.add_argument("--no-pretrained", default=True, action="store_false")
    parser.add_argument("--freeze-embeddings", default=False, action="store_true")
    parser.add_argument("--peft", default=False, action="store_true")
    parser.add_argument("--head-only", default=False, action="store_true")

    # rw specific
    parser.add_argument("--latent-size-scale", type=int, default=1)
    parser.add_argument("--reduced-dim", type=int, default=2)
    parser.add_argument("--dyna-num-heads", type=int, default=16)
    parser.add_argument("--dyna-concat", action="store_true", default=False)

    # memory blocks specific
    parser.add_argument("--memory-blocks", choices=["", "mlp", "conv"], default="")
    parser.add_argument("--memory-mlp-ratio", type=float, default=0.5)

    parser.add_argument("--memory-encoder-depths", nargs="+", type=int, default=[3, 3])
    parser.add_argument(
        "--memory-encoder-downsamples", nargs="+", type=int, default=[2, 1]
    )
    parser.add_argument("--memory-decoder-depths", nargs="+", type=int, default=[3, 3])

    # Augmentation arguments.
    parser.add_argument(
        "--augmentations",
        default=False,
        action="store_true",
        help="Use augmentations",
    )
    parser.add_argument(
        "--rand-aug",
        default=False,
        action="store_true",
        help="Use RandAugment",
    )
    parser.add_argument(
        "--num-ops",
        type=int,
        default=2,
        help="Number of RandAugment operations",
    )
    parser.add_argument(
        "--magnitude",
        type=int,
        default=15,
        help="RandAugment magnitude",
    )
    parser.add_argument(
        "--use-mixup",
        default=False,
        action="store_true",
        help="Use mixup",
    )
    parser.add_argument(
        "--use-cutmix",
        default=False,
        action="store_true",
        help="Use cutmix",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=1.0,
        help="Mixup alpha",
    )
    parser.add_argument("--random-erasing", default=False, action="store_true")
    parser.add_argument("--erase-p", type=float, default=0.25)
    parser.add_argument("--bce", default=False, action="store_true")

    # Other arguments.
    parser.add_argument("--checkpoints-path", type=Path, default=None)
    parser.add_argument("--checkpoint-profile", type=str, default="")
    parser.add_argument(
        "--checkpoint-per-epoch",
        default=False,
        action="store_true",
        help="Enable to checkpoint per epoch",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=None,
        help="Use a subset of the data for training",
    )
    parser.add_argument("--wandb", default=False, action="store_true")
    parser.add_argument("--gradnorm", default=False, action="store_true")

    return parser.parse_args()


class Model(L.LightningModule):
    def __init__(self, model, args):
        super().__init__()

        self.args = args
        self.model = model

        if args.bce:
            # self.criterion = BinaryCrossEntropy()
            self.criterion = BinaryCrossEntropy(sum_classes=True)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # EMA Student.
        self.ema_model = torch.optim.swa_utils.AveragedModel(
            self.model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(self.args.ema),
        )

        self.cutmix_or_mixup = []
        if args.use_mixup:
            self.cutmix_or_mixup.append(
                tvt.MixUp(
                    alpha=args.mixup_alpha,
                    num_classes=args.num_classes,
                )
            )
        if args.use_cutmix:
            self.cutmix_or_mixup.append(
                tvt.CutMix(
                    alpha=args.mixup_alpha,
                    num_classes=args.num_classes,
                )
            )
        if not (args.use_mixup or args.use_cutmix):
            self.cutmix_or_mixup = [tvt.Identity()]
        self.cutmix_or_mixup = tvt.RandomChoice(self.cutmix_or_mixup)

        self.random_erasing = tvt.Identity()
        if args.random_erasing:
            self.random_erasing = tvt.RandomErasing(p=args.erase_p)

        self.running_loss = 0
        self.highest_val_accuracy = float("-inf")
        self.highest_ema_val_accuracy = float("-inf")
        self.process_accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.process_accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )
        self.process_ema_accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.process_ema_accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )
        self.memory_accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.memory_accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )
        self.memory_ema_accuracy_top1 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=1
        )
        self.memory_ema_accuracy_top5 = Accuracy(
            "multiclass", num_classes=model.num_classes, top_k=5
        )

    def training_step(self, batch, batch_idx):
        x, label = batch
        x = self.random_erasing(x)
        x, label = self.cutmix_or_mixup(x, label)
        memory, process = self.model(x)
        loss = self.criterion(memory, label)
        loss += self.criterion(process, label)
        # Running loss.
        self.running_loss += loss.detach().item()
        self.log("train loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, label = batch
        memory, process = self.model(x)
        loss = self.criterion(memory, label)
        loss += self.criterion(process, label)

        # ema model eval
        ema_memory, ema_process = self.ema_model(x)
        ema_loss = self.criterion(ema_memory, label)
        ema_loss += self.criterion(ema_process, label)

        # Accuracy
        self.memory_accuracy_top1(memory, label)
        self.memory_accuracy_top5(memory, label)
        self.memory_ema_accuracy_top1(ema_memory, label)
        self.memory_ema_accuracy_top5(ema_memory, label)
        self.log(
            "memory_accuracy_top1",
            self.memory_accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "memory_accuracy_top5",
            self.memory_accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "memory_ema_accuracy_top1",
            self.memory_ema_accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "memory_ema_accuracy_top5",
            self.memory_ema_accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.process_accuracy_top1(process, label)
        self.process_accuracy_top5(process, label)
        self.process_ema_accuracy_top1(ema_process, label)
        self.process_ema_accuracy_top5(ema_process, label)
        self.log(
            "process_accuracy_top1",
            self.process_accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "process_accuracy_top5",
            self.process_accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "process_ema_accuracy_top1",
            self.process_ema_accuracy_top1,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "process_ema_accuracy_top5",
            self.process_ema_accuracy_top5,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log(
            "test loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "ema test loss",
            ema_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def _peft(self):
        keys_to_unfreeze = [
            "process_pos_embed",
            "memory_pos_embed",
            "process_embedder",
            "memory_embedder",
            "read_norm",
            "write_norm",
            "process_norm",
            "process_head",
            "head",
            "mem_blocks",
            "write_head",
            "read_head",
        ]
        for key, param in self.model.named_parameters():
            param.requires_grad = False
            for unfreeze_key in keys_to_unfreeze:
                if unfreeze_key in key:
                    print(f"Training: {key}")
                    param.requires_grad = True
                    break

    def configure_optimizers(self):
        if self.args.peft:
            self._peft()

        if self.args.freeze_embeddings:
            for name, param in self.model.named_parameters():
                if "embedder" in name:
                    print("Freezing: ", name)
                    param.requires_grad = False
                if "patch_embed" in name:
                    print("Freezing: ", name)
                    param.requires_grad = False

        lr = self.args.learning_rate
        # Determine parameters to optimize.
        parameters = list(self.model.parameters())

        if args.head_only:
            parameters = list(self.model.head.parameters()) + list(
                self.model.process_head.parameters()
            )

        if self.args.optimizer == "shampoo":
            optimizer = DistributedShampoo(
                parameters,
                lr=lr,
                betas=(0.9, 0.999),
                epsilon=1e-12,
                weight_decay=self.args.weight_decay,
                precondition_frequency=args.precondition_frequency,
                max_preconditioner_dim=args.max_preconditioner_dim,
                start_preconditioning_step=args.start_preconditioning_step,
                use_decoupled_weight_decay=True,
                grafting_config=AdamGraftingConfig(
                    beta2=0.999,
                    epsilon=1e-8,
                ),
            )
        elif self.args.optimizer == "adamw":
            optimizer = optim.AdamW(
                parameters,
                lr=lr,
                weight_decay=self.args.weight_decay,
            )

        print(
            f"Using {args.optimizer} optimizer with scheduler: ",
            self.args.scheduler,
            "for ",
            self.args.total_steps,
            "steps",
            "with warmup steps: ",
            self.args.warmup_steps,
        )
        if self.args.scheduler == "poly":
            scheduler = create_poly_with_warmup(
                optimizer=optimizer,
                total_epochs=self.args.total_steps,
                warmup_epochs=self.args.warmup_steps,
                lr=lr,
                warmup_lr=self.args.warmup_lr,
                power=self.args.power,
            )
        elif self.args.scheduler == "cosine":
            scheduler = create_cosine_with_warmup(
                optimizer=optimizer,
                total_epochs=self.args.total_steps,
                warmup_epochs=self.args.warmup_steps,
                lr=lr,
                warmup_lr=self.args.warmup_lr,
                min_lr=self.args.min_lr,
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        if self.args.gradnorm:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self.ema_model.update_parameters(self.model)

    def on_train_epoch_end(self) -> None:
        if args.dataset == "imagenet-21k":
            # Save Model
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing.pth"
                torch.save(self.model.state_dict(), save_path)
                torch.save(
                    self.ema_model.state_dict(),
                    (self.args.save_loc / f"best_ema.pth"),
                )

    def on_validation_epoch_end(self) -> None:
        acc = self.memory_accuracy_top1.compute()
        if acc > self.highest_val_accuracy:
            self.highest_val_accuracy = acc
            # Save Model
            if self.global_rank == 0:
                save_path = self.args.save_loc / f"best_performing.pth"
                torch.save(self.model.state_dict(), save_path)
                # torch.save(
                #     self.ema_model.state_dict(),
                #     (self.args.save_loc / f"best_ema.pth"),
                # )
        acc = self.memory_ema_accuracy_top1.compute()
        if acc > self.highest_ema_val_accuracy:
            self.highest_ema_val_accuracy = acc
            # Save Model
            if self.global_rank == 0:
                # save_path = self.args.save_loc / f"best_performing.pth"
                # torch.save(self.model.state_dict(), save_path)
                torch.save(
                    self.ema_model.state_dict(),
                    (self.args.save_loc / f"best_ema.pth"),
                )


def create_model(args, pretrained=True):
    model_name = args.model
    model_base, model_size = model_name.split("_")
    if model_base == "vit":
        model, config = vit_factory(
            model_size,
            pretrained=pretrained,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
    elif model_base == "deit":
        model = deit_factory(model_size, pretrained=pretrained)
    if model_base == "vittm":
        model = vittm_factory(
            model_size,
            pretrained=pretrained,
            memory_ps=args.memory_ps,
            process_ps=args.process_ps,
            latent_size_scale=args.latent_size_scale,
            rw_head_type=args.rw_head_type,
            fusion_type=args.fusion_type,
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
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
            num_classes=args.num_classes,
            drop_path_rate=args.drop_path_rate,
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
    return model


def main(args):
    # Setup W&B logger.
    if args.wandb:
        wandb_logger = WandbLogger(project="vttm-new")

    callbacks = []

    # Create lr monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Checkpoint setup
    if args.dataset == "imagenet-21k":
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_loc,
            filename="{epoch:02d}",
            save_on_train_epoch_end=True,
            save_weights_only=True if args.optimizer == "shampoo" else False,
        )
        callbacks.append(checkpoint_callback)
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=args.save_loc,
            save_top_k=2,
            monitor="test loss",
            filename="best-{epoch:02d}",
            auto_insert_metric_name=True,
            save_weights_only=True if args.optimizer == "shampoo" else False,
        )
        checkpoint_callback_every = ModelCheckpoint(
            dirpath=args.save_loc,
            filename="{epoch:02d}",
            save_on_train_epoch_end=True,
            save_weights_only=True if args.optimizer == "shampoo" else False,
        )
        callbacks.append(checkpoint_callback)
        callbacks.append(checkpoint_callback_every)

    trainer = L.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accumulations,
        max_epochs=args.epochs,
        overfit_batches=args.overfit_batches,
        gradient_clip_val=args.grad_clip,
        callbacks=[lr_monitor, *callbacks],
        log_every_n_steps=10,
        logger=wandb_logger if args.wandb else None,
        strategy=args.strategy,
        benchmark=True,  # cudnn benchmarking, allows for faster training.
        # enable_checkpointing=False,  # Disable automatic checkpointing (we do this manually).
    )

    if args.model_checkpoint:
        print("Loading from checkpoint: {args.model_checkpoint}")
        model = create_model(args, pretrained=False)
        state_dict = torch.load(args.model_checkpoint)
        if args.replace_head:
            # if "ema" in args.model_checkpoint.parts[-1]:
            #     state_dict.pop("module.head.weight")
            #     state_dict.pop("module.head.bias")
            #     state_dict.pop("module.process_head.weight")
            #     state_dict.pop("module.process_head.bias")
            # else:
            state_dict.pop("head.weight")
            state_dict.pop("head.bias")
            state_dict.pop("process_head.weight")
            state_dict.pop("process_head.bias")
        model.load_state_dict(state_dict, strict=False)
    else:
        model = create_model(args, pretrained=args.no_pretrained)

    if args.flexi_process:
        if "ttm" in args.model and not args.model_checkpoint:
            model = apply_flexi_patch_embed(
                model=model,
                process_embedding_name="memory_embedder.proj.weight",
                patch_embedding_name="patch_embed.proj.weight",
                new_patch_size=(args.memory_ps, args.memory_ps),
            )
            model = apply_flexi_patch_embed(
                model=model,
                process_embedding_name="process_embedder.proj.weight",
                patch_embedding_name="patch_embed.proj.weight",
                new_patch_size=(args.process_ps, args.process_ps),
            )
    if args.compile:
        model = torch.compile(model)

    train_dataset, test_dataset = create_dataset(args)
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if args.overfit_batches else True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    val_loader = None
    if test_dataset:
        val_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            num_workers=4,
            pin_memory=args.pin_memory,
        )

    args.total_steps = (
        len(loader)
        * args.epochs
        // (args.accumulations * args.devices * args.num_nodes)
    )

    model = Model(model, args)

    if trainer.global_rank == 0:
        if args.wandb:
            wandb_logger.experiment.config.update(
                {
                    **vars(args),
                }
            )
        save_loc.mkdir(parents=True, exist_ok=True)

    # Trainer Fit.
    trainer.fit(
        model,
        train_dataloaders=loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_checkpoint,
    )


def create_directory(save_base_path, now, i=0):
    new_now = now + f"-{i}"
    save_loc = save_base_path / new_now
    if save_loc.exists():
        return create_directory(save_base_path, now, i + 1)
    return save_loc


if __name__ == "__main__":
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S.%f")

    ### In case we want to add a more descriptive checkpoint dir folder
    if args.checkpoint_profile != "":
        now = now + "-" + args.checkpoint_profile.replace(" ", "-").replace("_", "-")

    save_base_path = (
        args.checkpoints_path if args.checkpoints_path else DEFAULT_CHECKPOINTS_PATH
    )

    # save_loc = create_directory(save_base_path, now)

    # save_loc.mkdir(parents=True, exist_ok=False)
    save_loc = save_base_path / now

    args.save_loc = save_loc
    args.pretraining = False

    print(f"train_lt.py: Checkpoint folder location: {save_loc}")

    main(args)
