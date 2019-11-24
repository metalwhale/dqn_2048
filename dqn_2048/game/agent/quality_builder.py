"""
Quality builder
"""

from ...base import QualityBuilder as BaseQualityBuilder
from .quality import Quality

class QualityBuilder(BaseQualityBuilder):
    """
    Quality builder
    """

    def build(self) -> Quality:
        pass
