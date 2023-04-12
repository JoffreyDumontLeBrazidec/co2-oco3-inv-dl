# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
from tensorflow import keras
import tensorflow as tf
from dataclasses import dataclass, field


def standard_CNN(input_shape, Nclasses):
    """Build standard CNN model for regression."""

    inputs = keras.layers.Input(input_shape)

    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(inputs)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", strides=1)(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="valid", strides=2)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(Nclasses)(x)
    outputs = keras.layers.LeakyReLU(alpha=0.3)(x)

    model = keras.Model([inputs], outputs)

    return model


@dataclass
class Keras_reg_model_builder:
    name: str
    input_shape: list
    classes: int
    init_w: str = None
    drop_rate: np.float32 = 0.2

    def __post_init__(self):
        if self.init_w == "random":
            self.init_w = None

    def get_model(self):
        """Return keras regression model."""
        base_model = self.define_base()
        model = self.add_top(base_model)
        return model

    def define_base(self):
        """Define base of the model with keras API."""
        model_to_call = getattr(keras.applications, self.name)
        base_model = model_to_call(
            include_top=False,
            weights=self.init_w,
            input_shape=self.input_shape,
            drop_connect_rate=self.drop_rate
        )
        base_model.trainable = True
        return base_model

    def add_top(self, base_model):
        """Add top layers to keras regression model."""
        inputs = keras.layers.Input(shape=self.input_shape, name="input_layer")
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
        x = keras.layers.Dense(self.classes, name="output_layer")(x)
        outputs = keras.layers.LeakyReLU(
            alpha=0.3, dtype=tf.float32, name="activation_layer"
        )(x)
        model = keras.Model([inputs], outputs)
        return model


@dataclass
class Reg_model_builder:
    """Return appropriate regression model."""

    name: str = "EfficientNetB0"
    input_shape: np.ndarray = field(default_factory=lambda: [160, 160, 3])
    classes: int = 1
    init_w: str = None
    drop_rate: np.float32 = 0.2

    def get_model(self):
        """Return regression model, keras or locals."""
        if (
            self.name.startswith("EfficientNet")
            or self.name.startswith("ResNet")
            or self.name.startswith("DenseNet")
        ):
            keras_builder = Keras_reg_model_builder(
                self.name, self.input_shape, self.classes, self.init_w, self.drop_rate
            )
            model = keras_builder.get_model()

        else:
            model_to_call = locals()[self.name]
            model = standard_CNN(self.input_shape, self.classes)

        return model
