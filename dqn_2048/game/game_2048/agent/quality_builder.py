"""
Quality builder
"""

from typing import Callable

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Optimizer

from ...base import QualityBuilder as BaseQualityBuilder
from .quality import Quality

class QualityBuilder(BaseQualityBuilder):
    """
    Quality builder
    """

    def __init__(self):
        self.gamma = 0.0
        self.output_size = 0
        self.model_builder = None
        self.optimizer = None
        self.delta_clip = np.inf

    def set_gamma(self, gamma: float):
        """
        # Arguments
            gamma: float. The discount factor, used for Bellman approximation.
        """
        self.gamma = gamma
        return self

    def set_output_size(self, output_size: int):
        """
        # Arguments
            output_size: int. Output size of the model.
        """
        self.output_size = output_size
        return self

    def set_model_builder(self, model_builder: Callable[[int], Model]):
        """
        # Arguments
            model_builder: Callable[[int], Model]. Model builder.
        """
        self.model_builder = model_builder
        return self

    def set_optimizer(self, optimizer: Optimizer):
        """
        # Arguments
            optimizer: Optimizer. Optimizer.
        """
        self.optimizer = optimizer
        return self

    def set_delta_clip(self, delta_clip: float):
        """
        # Arguments
            delta_clip: float. Clip value.
        """
        self.delta_clip = delta_clip
        return self

    def build(self) -> Quality:
        return Quality(
            self.gamma, self.output_size,
            self.model_builder, self.optimizer,
            delta_clip=self.delta_clip
        )
