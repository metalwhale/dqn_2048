"""
Action
"""

from __future__ import annotations

from abc import abstractstaticmethod

class Action:
    """
    Action
    """

    @abstractstaticmethod
    def random() -> Action:
        """
        # Returns random action.
        """
