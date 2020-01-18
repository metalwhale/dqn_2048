"""
Quality
"""

from __future__ import annotations

from datetime import datetime
from os import makedirs, path
from typing import Callable, List

from numpy import array, inf, isinf, ndarray, random, zeros
from keras import Model
from keras import backend as K
from keras.layers import Input, Lambda
from keras.optimizers import Optimizer
from tensorflow import where

from ...base import Action, Quality as BaseQuality, Experience
from ..environment.state import State

class Quality(BaseQuality):
    """
    Quality
    """

    def __init__(
            self,
            gamma: float, output_size: int,
            model_builder: Callable[[int], Model], optimizer: Optimizer,
            delta_clip: float = inf
        ):
        """
        # Arguments
            gamma: float. Gamma.
            output_size: int. Output size of the model.
            model_builder: Callable[[int], Model]. Takes output size as param and returns model.
            optimizer: Optimizer. Optimizer used when training model.
            delta_clip: float. Used for calculating loss.
        """
        super().__init__(gamma, output_size)
        self.delta_clip = delta_clip
        self._model = model_builder(self.output_size)
        # Create learning model which is actually used for training
        # For more details, see https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py
        self._learning_model = self._create_learning_model(
            self._model, self.output_size, optimizer
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

    def _predict(self, state: State) -> ndarray:
        return self._model.predict(array([state.data]))[0]

    def _create_learning_model(self, model: Model, output_size: int, optimizer: Optimizer) -> Model:
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
        learning_model.compile(optimizer, loss=losses)
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
