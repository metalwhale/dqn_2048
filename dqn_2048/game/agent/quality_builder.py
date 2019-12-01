"""
Quality builder
"""

from numpy import inf

from ...base import QualityBuilder as BaseQualityBuilder
from .quality import Quality

class QualityBuilder(BaseQualityBuilder):
    """
    Quality builder
    """

    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.gamma = 0.0
        self.delta_clip = inf
        self.learning_rate = 0.0

    def set_input_size(self, input_size: int):
        """
        # Arguments
            input_size: int. Input size of the model.
        """
        self.input_size = input_size
        return self

    def set_output_size(self, output_size: int):
        """
        # Arguments
            output_size: int. Output size of the model.
        """
        self.output_size = output_size
        return self

    def set_gamma(self, gamma: float):
        """
        # Arguments
            gamma: float. The discount factor, used for Bellman approximation.
        """
        self.gamma = gamma
        return self

    def set_delta_clip(self, delta_clip: float):
        """
        # Arguments
            delta_clip: float. Clip value.
        """
        self.delta_clip = delta_clip
        return self

    def set_learning_rate(self, learning_rate: float):
        """
        # Arguments
            learning_rate: float. Learning rate of the optimizer.
        """
        self.learning_rate = learning_rate
        return self

    def build(self) -> Quality:
        return Quality(
            self.gamma, self.input_size, self.output_size,
            self.delta_clip, self.learning_rate
        )
