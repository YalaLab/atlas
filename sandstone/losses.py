from typing import Tuple, Dict
from abc import ABC, abstractmethod
from collections import OrderedDict

from torch import Tensor
import torch.nn.functional as F
from timm import loss


class AbstractLoss(ABC):
    """Abstract base class for all loss implementations."""
    
    def __init__(self, args, **kwargs):
        self.args = args

    @abstractmethod
    def __call__(self, **kwargs) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Calculate the loss value.
        
        Returns:
            Tuple containing:
                - Loss tensor
                - Dictionary of logging values
        """
        pass


class CrossEntropyLoss(AbstractLoss):
    """Standard cross-entropy loss implementation."""
    
    def __call__(self, logit=None, target=None, label_smoothing=0.0, **extras):
        logging_dict = OrderedDict()
        loss_value = F.cross_entropy(logit, target, label_smoothing=label_smoothing)
        logging_dict["cross_entropy_loss"] = loss_value.detach()
        return loss_value, logging_dict


class SoftTargetCrossEntropyLoss(AbstractLoss):
    """Cross entropy loss that accepts soft targets."""
    
    def __init__(self, args):
        super().__init__(args)
        self.loss_fn = loss.SoftTargetCrossEntropy()

    def __call__(self, logit=None, target=None, **extras):
        logging_dict = OrderedDict()
        loss_value = self.loss_fn(logit, target)
        logging_dict["soft_target_cross_entropy_loss"] = loss_value.detach()
        return loss_value, logging_dict