"""
Quality
"""

from __future__ import annotations

from typing import List, Tuple

from numpy import array, argmax
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras import backend as K

from ...base import Quality as BaseQuality
from ..environment.state import State
from ..environment.direction import Direction
from ..environment.action import Action
from .experience import Experience

class Quality(BaseQuality):
    """
    Quality.
    """

    def __init__(self, gamma: float, input_size: int, output_size: int, learning_rate: float):
        """
        # Arguments
            gamma: float. Gamma.
            input_size: int. Input size of the model.
            output_size: int. Output size of the model.
            learning_rate: float. Learning rate of the optimizer.
        """
        super().__init__(gamma)
        self.model = self.__create_model(input_size, output_size)
        self.model.compile(SGD(learning_rate), loss=self.__loss, metrics=["accuracy"])

    def learn(self, batch: List[Experience]):
        state_list: List[State] = [experience.state for experience in batch]
        state_data = [state.data for state in state_list]
        action_data = []
        for experience in batch:
            action: Action = experience.action
            data = [0] * action.space_size()
            data[action.data] = experience.value
            action_data.append(data)
        self.model.fit(array(state_data), array(action_data), verbose=0)

    def copied(self, training_quality: Quality):
        self.model.set_weights(training_quality.model.get_weights())

    def _select_action(self, state: State) -> Tuple[Action, float]:
        values = self.model.predict(array([state.data]))[0]
        max_index = argmax(values)
        return (Action(Direction(max_index)), values[max_index])

    @staticmethod
    def __create_model(input_size: int, output_size: int) -> Sequential:
        """
        # Arguments
            input_size: int. Input size.
            output_size: int. Output size.
        # Returns the model.
        """
        model = Sequential()
        model.add(Dense(16, activation="relu", input_shape=(input_size,)))
        model.add(Dense(12, activation="relu"))
        model.add(Dense(output_size, activation="softmax"))
        return model

    @staticmethod
    def __loss(y_true: List[float], y_pred: List[float]) -> float:
        """
        Calculate loss: L = (Qs,a - y) ^ 2
        """
        mask = K.cast(y_true != 0.0, "float32")
        loss = K.sum((y_pred - y_true) * mask)
        loss **= 2
        return loss
