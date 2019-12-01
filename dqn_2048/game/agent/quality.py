"""
Quality
"""

from __future__ import annotations

from datetime import datetime
from os import makedirs, path
from typing import List, Tuple

from numpy import array, argmax, isinf, ndarray, random, zeros
from keras import Model
from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Sequential
from keras.optimizers import SGD
from tensorflow import where

from ...base import Quality as BaseQuality
from ..environment.state import State
from ..environment.direction import Direction
from ..environment.action import Action
from .experience import Experience

class Quality(BaseQuality):
    """
    Quality
    """

    def __init__(
            self,
            gamma: float, input_size: int, output_size: int,
            delta_clip: float, learning_rate: float
        ):
        """
        # Arguments
            gamma: float. Gamma.
            input_size: int. Input size of the model.
            output_size: int. Output size of the model.
            delta_clip: float. Used for calculating loss.
            learning_rate: float. Learning rate of the optimizer.
        """
        super().__init__(gamma)
        self.output_size = output_size
        self.delta_clip = delta_clip
        self._model = self._create_model(input_size, output_size)
        # Create learning model which is actually used for training
        # For more details, see https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
        self._learning_model = self._create_learning_model(
            self._model, output_size, learning_rate
        )

    def learn(self, batch: List[Experience]):
        state_data = []
        targets = zeros((len(batch), self.output_size))
        masks = zeros((len(batch), self.output_size))
        dummies = random.rand(len(batch)) # Useless data, leaves the loss computation to lambda
        for i, experience in enumerate(batch):
            state: State = experience.state
            action: Action = experience.action
            value = experience.value
            state_data.append(state.data)
            targets[i][action.data] = value
            masks[i][action.data] = 1.0
        state_data = array(state_data)
        targets = array(targets).astype("float")
        masks = array(masks).astype("float")
        self._learning_model.train_on_batch([state_data, targets, masks], [dummies, targets])

    def copied(self, training_quality: Quality):
        self._model.set_weights(training_quality.weights)

    def save(self, dir_path: str):
        makedirs(dir_path, exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._model.save_weights(path.join(dir_path, f"{now}.hdf5"))

    @property
    def weights(self) -> List[ndarray]:
        """
        # Returns the weights of the model.
        """
        return self._model.get_weights()

    def _select_action(self, state: State) -> Tuple[Action, float]:
        values = self._model.predict(array([state.data]))[0]
        max_index = argmax(values)
        return (Action(Direction(max_index)), values[max_index])

    @staticmethod
    def _create_model(input_size: int, output_size: int) -> Model:
        """
        # Arguments
            input_size: int. Input size.
            output_size: int. Output size.
        # Returns the model.
        """
        model = Sequential()
        model.add(Dense(64, activation="relu", input_shape=(input_size,)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(output_size))
        model.compile("sgd", loss="mse")
        return model

    def _create_learning_model(self, model: Model, output_size: int, learning_rate: float) -> Model:
        """
        # Arguments
            model: Model. The main model to be "wrapped in".
            output_size: int. Output size.
            learning_rate: float. Optimizer's learning rate.
        # Returns learning model.
        """
        y_pred = model.output
        y_true = Input(shape=(output_size,))
        mask = Input(shape=(output_size,))
        loss_out = Lambda(self._clipped_masked_error, output_shape=(1,))([y_true, y_pred, mask])
        learning_model = Model(inputs=[model.input, y_true, mask], outputs=[loss_out, y_pred])
        losses = [
            lambda y_true, y_pred: y_pred,
            lambda y_true, y_pred: K.zeros_like(y_true)
        ]
        learning_model.compile(SGD(learning_rate), loss=losses)
        return learning_model

    def _clipped_masked_error(self, args):
        """
        See https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
        """
        y_true, y_pred, mask = args
        loss = self._huber_loss(y_true, y_pred, self.delta_clip)
        loss *= mask
        return K.sum(loss, axis=-1)

    @staticmethod
    def _huber_loss(y_true, y_pred, clip_value):
        """
        See https://github.com/keras-rl/keras-rl/blob/master/rl/util.py
        """
        diff = y_true - y_pred
        if isinf(clip_value):
            return 0.5 * K.square(diff)
        condition = K.abs(diff) < clip_value
        squared_loss = 0.5 * K.square(diff)
        linear_loss = clip_value * (K.abs(diff) - 0.5 * clip_value)
        return where(condition, squared_loss, linear_loss)
