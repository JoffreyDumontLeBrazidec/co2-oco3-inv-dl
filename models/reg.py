# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import sys
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf
from icecream import ic
from tensorflow import keras

from models.my_efficientnet import EfficientNet
from models.my_essential_inversors import (
    CCT,
    attention_cnn,
    deep_cnn_advanced,
    densenet_like_model,
    essential_regressor,
    inception_like_model,
    linear_regressor,
    simple_resnet,
    unet_like_model,
)
from models.my_mobilenet import MobileNet
from models.my_shufflenet import ShuffleNet
from models.my_squeezenet import SqueezeNet
from models.preprocessing import (
    CloudsLayer,
    ConditionalNoiseLayer,
    TrainingTimeNormalization,
)


def get_top_layers(classes: int, choice_top: str = "linear"):
    """Return top layers for regression model."""

    def top_layers(x):
        if choice_top in [
            "efficientnet",
            "squeezenet",
            "nasnet",
            "mobilenet",
            "shufflenet",
        ]:
            x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
            x = tf.keras.layers.Dense(classes, name="regressor")(x)
            outputs = tf.keras.layers.LeakyReLU(
                alpha=0.3, dtype=tf.float32, name="regressor_activ"
            )(x)
        elif choice_top == "linear":
            outputs = tf.keras.layers.Dense(classes, name="regressor")(x)
        else:
            x = tf.keras.layers.Dense(1)(x)
            outputs = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
        return outputs

    return top_layers


def get_core_model(
    name: str,
    input_shape: list,
    classes: int = 1,
    dropout_rate: float = 0.2,
):
    """Get core model for regression model."""
    if name == "efficientnet":
        core_model = EfficientNet(
            scaling_coefficient=1,
            input_shape=input_shape,
            classes=classes,
            dropout_rate=dropout_rate,
        )
    elif name == "linear":
        core_model = linear_regressor(input_shape)
    elif name == "essential":
        core_model = essential_regressor(input_shape, dropout_rate)
    elif name == "squeezenet":
        core_model = SqueezeNet(input_shape, dropout_rate, compression=0.4)
    elif name == "mobilenet":
        core_model = MobileNet(input_shape, scaling_coeff=0.4)
    elif name == "shufflenet":
        core_model = ShuffleNet(input_shape, scaling_coefficient=0.75)
    elif name == "CCT":
        core_model = CCT(input_shape)
    elif name == "densenet_like_model":
        core_model = densenet_like_model(input_shape)
    elif name == "unet_like_model":
        core_model = unet_like_model(input_shape)
    elif name == "inception_like_model":
        core_model = inception_like_model(input_shape)
    elif name == "deep_cnn_advanced":
        core_model = deep_cnn_advanced(input_shape)
    elif name == "attention_cnn":
        core_model = attention_cnn(input_shape)
    elif name == "simple_resnet":
        core_model = simple_resnet(input_shape)

    else:
        logging.info(f"Wrong model name selected: {name}")
        sys.exit()

    return core_model


@dataclass
class Reg_model_builder:
    """Return appropriate regression model."""

    name: str
    input_shape: list = field(default_factory=lambda: [64, 64, 3])
    classes: int = 1
    norm_layer: TrainingTimeNormalization = TrainingTimeNormalization(
        axis=-1, name="preproc_norm"
    )
    noise_layer: ConditionalNoiseLayer = ConditionalNoiseLayer([])
    dropout_rate: float = 0.2
    cloud_layer: CloudsLayer = CloudsLayer(np.zeros((1)), [])
    timedate_vector_size: int = 0

    def get_model(self):
        """Return regression model, keras or locals."""

        self.core_model = get_core_model(
            self.name,
            self.input_shape,
            self.classes,
            self.dropout_rate,
        )
        self.top_layers = get_top_layers(self.classes, self.name)

        inputs = tf.keras.layers.Input(self.input_shape, name="input_layer")
        x = self.core_model(inputs)

        if self.timedate_vector_size > 0:
            timedate_input = tf.keras.layers.Input(
                shape=self.timedate_vector_size, name="timedate_input_layer"
            )
            concatenated = tf.keras.layers.Concatenate()([x, timedate_input])
            x = tf.keras.layers.Dense(16, activation="elu")(concatenated)
            outputs = self.top_layers(x)

        else:
            outputs = self.top_layers(x)

        model = tf.keras.Model(inputs, outputs)
        return model
