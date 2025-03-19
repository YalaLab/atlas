import math
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LRScheduler,
    ReduceLROnPlateau,
)


class CosineAnnealingWarmup(LRScheduler):
    """
    Cosine annealing scheduler with warmup period.
    
    During warmup, the learning rate increases linearly from 0 to base_lr.
    After warmup, the learning rate follows a cosine decay from base_lr to min_lr.
    """
    
    def __init__(
        self,
        optimizer,
        max_epochs,
        warmup_epochs,
        min_lr=0,
        last_epoch=-1,
        verbose=False,
        **extras,
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            max_epochs: Total number of epochs
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
            last_epoch: The index of the last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Use (last_epoch + 1)/(warmup_epochs + 1) to avoid zero learning rate
            return [
                base_lr * ((self.last_epoch + 1) / (self.warmup_epochs + 1))
                for base_lr in self.base_lrs
            ]
        else:
            return [
                self.min_lr
                + (base_lr - self.min_lr)
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                    )
                )
                for base_lr in self.base_lrs
            ]


class BaseStepScheduler:
    """Base class for step-based schedulers that don't inherit from PyTorch's LRScheduler."""
    
    def state_dict(self):
        """Return the state of the scheduler as a dict."""
        return {}

    def load_state_dict(self, state_dict):
        """Load the scheduler state.
        
        Args:
            state_dict: Scheduler state. Should be an empty dict for BaseStepScheduler.
        """
        if state_dict:
            raise ValueError(
                f"state_dict is not empty but step scheduler does not support "
                f"loading non-empty state dict: {state_dict.keys()}"
            )


class StepCosineAnnealingWarmup(BaseStepScheduler):
    """
    Step-based cosine annealing scheduler with warmup period.
    
    This scheduler manually adjusts the learning rate per step/epoch rather than
    being called automatically by the PyTorch training loop.
    """
    
    def __init__(
        self,
        optimizer,
        max_epochs,
        lr,
        warmup_epochs,
        min_lr=0
    ):
        """
        Args:
            optimizer: Wrapped optimizer
            max_epochs: Total number of epochs
            lr: Base learning rate
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
        """
        self.optimizer = optimizer
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.epochs = max_epochs

    def adjust_learning_rate(self, epoch):
        """
        Adjust the learning rate based on the current epoch.
        
        During warmup, the learning rate increases linearly from 0 to base_lr.
        After warmup, the learning rate follows a cosine decay from base_lr to min_lr.
        
        Args:
            epoch: Current epoch (can be a float for per-step scheduling)
            
        Returns:
            Current learning rate
        """
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )

        for param_group in self.optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

        return lr