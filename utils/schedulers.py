import torch
import torch.optim as optim
from typing import List, Tuple
import math
import warnings


def create_linear_warmup(optimizer, lr, warmup_lr, warmup_epochs):
    return optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_lr / lr, end_factor=1, total_iters=warmup_epochs
    )


def create_cosine_with_warmup(
    optimizer, total_epochs, warmup_epochs, lr, warmup_lr, min_lr=0.0
):
    warmup_schedule = create_linear_warmup(optimizer, lr, warmup_lr, warmup_epochs)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=min_lr
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_schedule, cosine_schedule], milestones=[warmup_epochs]
    )

def create_poly_with_warmup(optimizer, total_epochs, warmup_epochs, lr, warmup_lr, power=0.9):
    warmup_schedule = create_linear_warmup(optimizer, lr, warmup_lr, warmup_epochs)
    poly_schedule = optim.lr_scheduler.PolynomialLR(
        optimizer, total_epochs - warmup_epochs, power=power
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, [warmup_schedule, poly_schedule], milestones=[warmup_epochs]
    )


class CosineAnnealingWithWarmup(optim.lr_scheduler.CosineAnnealingLR):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        eta_min: float = 0,
        warmup_lr: float = 0,
        last_epoch: int = -1,
        verbose: bool = "deprecated",
        warmup_epochs: int = 10,
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        T_max = T_max - warmup_epochs  # These are the annealing epochs.
        super().__init__(optimizer, T_max, eta_min, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        if self.last_epoch < self.warmup_epochs:
            return self._warmup_lr(self.last_epoch)
        else:
            return self._cosine_lr(self.last_epoch - self.warmup_epochs)

    # This is nearly-identical to PyTorch implementation, but last epoch is now an argument.
    def _warmup_lr(self, last_epoch) -> List[float]:
        if last_epoch == 0:
            return [self.warmup_lr for group in self.optimizer.param_groups]
        return [
            self.warmup_lr
            + last_epoch * (base_lr - self.warmup_lr) / self.warmup_epochs
            for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
        ]

    # This is nearly-identical to PyTorch implementation, but last epoch is now an argument.
    def _cosine_lr(self, last_epoch) -> List[float]:
        if last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]
        elif self._step_count == 1 and last_epoch > 0:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif (last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        return [
            (1 + math.cos(math.pi * last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]
