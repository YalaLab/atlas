from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import torch
import numpy as np
from torch import Tensor
from torchmetrics.functional import (
    accuracy,
    auc,
    auroc,
    average_precision,
    f1_score,
    precision_recall,
    precision_recall_curve,
)


class AbstractMetric(ABC):
    """Abstract base class for all metric implementations."""
    
    def __init__(self, args, **kwargs):
        self.args = args

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, Tensor]:
        """
        Calculate metric values.
        
        Returns:
            Dictionary of metric names and values
        """
        pass


class Accuracy(AbstractMetric):
    """Simple accuracy metric implementation."""
    
    @property
    def metric_keys(self):
        """Required input keys for this metric."""
        return ["preds", "golds"]

    def __call__(self, preds=None, golds=None, **extras):
        metrics = OrderedDict()
        metrics["accuracy"] = accuracy(golds, preds)
        return metrics

