# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from hydra import compose, initialize
from hydra.utils import call, instantiate
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import include.generators as generators
from Data import Data_train


@dataclass
class Trainer:
    """Train Convolutional Neural Networks models."""

    generator: generators.InvDataGen
    callbacks: list = field(default_factory=lambda: [])
    batch_size: int = 32
    max_epochs: int = 10

    def train_model(self, model: tf.keras.Model, data: Data_train) -> tf.keras.Model:
        """Train model and evaluate validation."""
        self.history = model.fit(
            self.generator,
            epochs=self.max_epochs,
            validation_data=(data.x.valid, data.y.valid),
            verbose="auto",
            steps_per_epoch=int(
                np.floor(data.x.xco2_plumes_train.shape[0] / self.batch_size)
            ),
            callbacks=self.callbacks,
            shuffle=True,
        )

        return model

    def get_history(self):
        """Return history."""
        return self.history

    def get_val_loss(self):
        """Return best val loss."""
        return np.min(self.history.history["val_loss"])
