"""
Quality
"""

from typing import List, Tuple

from ...base import Quality as BaseQuality
from ...base import Experience
from ..environment.state import State
from ..environment.action import Action

class Quality(BaseQuality):
    """
    Quality.
    """

    def learn(self, batch: List[Experience]):
        pass

    def copied(self, training_quality: Quality):
        pass

    def _select_action(self, state: State) -> Tuple[Action, float]:
        pass
