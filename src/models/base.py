"""Abstract base class for all prediction models (GAN, supervised, classifier, etc.)."""
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch


class PredictionModel(ABC):
    """Common interface for all model types.

    Every model must support:
    - train_epoch(train_loader) -> dict with at least "loss" key
    - predict(xb: Tensor) -> (y_reg, y_cls_logits, y_quantiles) on CPU
    - step_schedulers() -> advance LR schedulers
    - state_dict property for early stopping checkpoint
    """

    @abstractmethod
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """One epoch of training. Returns dict with loss values."""

    @abstractmethod
    def predict(self, xb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference. Returns (y_reg [B], y_cls [B], y_q [B,Q]) on CPU.

        Models that don't produce all three outputs should return zeros for unused ones.
        """

    def step_schedulers(self):
        """Advance LR schedulers. No-op if not applicable."""

    @abstractmethod
    def get_state_dict(self) -> dict:
        """Return model state for checkpointing (early stopping)."""

    @abstractmethod
    def load_state_dict(self, state: dict):
        """Restore model state from checkpoint."""
