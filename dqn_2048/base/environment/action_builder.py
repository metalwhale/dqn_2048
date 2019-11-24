"""
Action builder
"""

from abc import abstractstaticmethod

from .action import Action

class ActionBuilder:
    """
    Action builder
    """

    @abstractstaticmethod
    def randomly_build() -> Action:
        """
        # Returns randomly action.
        """
