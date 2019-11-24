"""
Action builder
"""

from random import choice

from ...base import ActionBuilder as BaseActionBuilder
from .direction import Direction
from .action import Action

class ActionBuilder(BaseActionBuilder):
    """
    Action builder
    """

    @staticmethod
    def randomly_build() -> Action:
        return Action(choice(list(Direction)))
