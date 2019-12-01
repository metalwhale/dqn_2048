"""
Quality builder
"""

from abc import abstractmethod

from .quality import Quality

class QualityBuilder:
    """
    Quality builder
    """

    @abstractmethod
    def build(self) -> Quality:
        """
        # Returns new quality.
        """
