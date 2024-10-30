# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling1D,
    Input,
    Lambda,
    LayerNormalization,
    MaxPooling2D,
    MultiHeadAttention,
    Multiply,
    Reshape,
    Subtract,
    concatenate,
)
from tensorflow.keras.models import Model


def linear_regressor(input_shape: list):
    """Linear regressor."""
    core_model = tf.keras.models.Sequential()
    core_model.add(tf.keras.Input(shape=input_shape))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    core_model.add(tf.keras.layers.Conv2D(16, (3, 3)))
    core_model.add(tf.keras.layers.Flatten())
    core_model.add(tf.keras.layers.Dense(16))
    return core_model


def essential_regressor(input_shape: list, dropout_rate: float = 0.2):
    """Essential regressor."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(dropout_rate / 2)(x)
    if input_shape[1] == 64:
        x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate * 3 / 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(dropout_rate * 3 / 2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.Flatten()(x)
    core_model = tf.keras.Model(inputs, x)
    return core_model


def residual_block(x, filters, kernel_size=(3, 3), strides=1):
    shortcut = x
    x = Conv2D(filters, kernel_size, padding="same", strides=strides)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding="same")(
        shortcut
    )
    shortcut = tf.keras.layers.Dropout(0.3)(shortcut)
    x = Add()([x, shortcut])
    x = Activation("elu")(x)
    return x


def simple_resnet(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = residual_block(x, 64)
    x = residual_block(x, 64, strides=2)
    x = residual_block(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def attention_module(x, filters):
    p = 1  # Pooling size
    t = 2  # Transformation ratio
    r = 1  # Number of attention modules

    x = Conv2D(filters * t, kernel_size=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = Conv2D(filters, kernel_size=1)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Activation("sigmoid")(x)

    return x


def attention_cnn(input_shape):
    inputs = Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    attention = attention_module(x, 32)
    attention = attention_module(attention, 32)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    attention = attention_module(x, 64)
    attention = attention_module(attention, 64)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    attention = attention_module(x, 64)
    attention = attention_module(attention, 64)
    x = Multiply()([x, attention])
    x = tf.keras.layers.MaxPooling2D()(x)

    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def deep_cnn_advanced(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = Conv2D(64, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation="elu", strides=1)(x)
    x = Conv2D(128, (3, 3), activation="elu", strides=1)(x)
    x = Dropout(0.4)(x)
    x = MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.AveragePooling2D()(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def inception_module(x, filters):
    branch1 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)

    branch2 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)
    branch2 = Conv2D(filters, (3, 3), activation="elu", padding="same")(branch2)

    branch3 = Conv2D(filters, (1, 1), activation="elu", padding="same")(x)
    branch3 = Conv2D(filters, (5, 5), activation="elu", padding="same")(branch3)

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch4 = Conv2D(filters, (1, 1), activation="elu", padding="same")(branch4)

    x = concatenate([branch1, branch2, branch3, branch4], axis=-1)
    return x


def inception_like_model(input_shape):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding="valid", strides=2)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = inception_module(inputs, 16)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = inception_module(x, 32)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = MaxPooling2D((2, 2))(x)

    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="elu", strides=1)(x)
    x = tf.keras.layers.AveragePooling2D()(x)
    x = Flatten()(x)
    model = Model(inputs, x)
    return model


def conv_block(x, filters):
    x = Conv2D(filters, (3, 3), activation="elu", padding="same")(x)
    x = Conv2D(filters, (3, 3), activation="elu", padding="same")(x)
    return x


def unet_like_model(input_shape):
    inputs = Input(shape=input_shape)

    c1 = conv_block(inputs, 16)  # Reduced from 32 to 16
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)  # Reduced from 64 to 32
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)  # Reduced from 128 to 64

    u4 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c3)
    u4 = concatenate([u4, c2])
    c4 = conv_block(u4, 32)

    u5 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c4)
    u5 = concatenate([u5, c1])
    c5 = conv_block(u5, 8)

    x = Conv2D(32, (3, 3), activation="elu", padding="same")(
        c5
    )  # Adjust the number of filters

    # Average Pooling to reduce spatial dimensions
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)

    model = Model(inputs, x)
    return model


def dense_block(x, growth_rate, layers_in_block):
    for _ in range(layers_in_block):
        cb = Conv2D(growth_rate, (3, 3), padding="same")(x)
        cb = Dropout(0.2)(cb)
        cb = BatchNormalization()(cb)
        cb = Activation("elu")(cb)
        x = Concatenate(axis=-1)([x, cb])
    return x


def transition_layer(x, compression_factor=0.5):
    reduced_filters = int(x.shape[-1] * compression_factor)
    x = Conv2D(reduced_filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)
    x = AveragePooling2D((2, 2))(x)
    return x


def densenet_like_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("elu")(x)

    x = dense_block(x, 16, 3)
    x = transition_layer(x)

    x = dense_block(x, 16, 3)
    x = transition_layer(x)

    x = dense_block(x, 32, 3)
    x = transition_layer(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="elu", strides=1)(x)

    x = Flatten()(x)

    model = Model(inputs, x)
    return model


## TRANSFORMER

from tensorflow import keras
from tensorflow.keras import layers

positional_emb = True
conv_layers = 2

transformer_layers = 2
stochastic_depth_rate = 0.1


class CCTTokenizer(layers.Layer):
    def __init__(
        self,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=3,
        pooling_stride=2,
        num_conv_layers=conv_layers,
        num_output_channels=[64, 64],
        positional_emb=positional_emb,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # This is our tokenizer.
        self.conv_model = keras.Sequential()
        for i in range(num_conv_layers):
            self.conv_model.add(
                layers.Conv2D(
                    num_output_channels[i],
                    kernel_size,
                    stride,
                    padding="valid",
                    use_bias=False,
                    activation="relu",
                    kernel_initializer="he_normal",
                )
            )
            self.conv_model.add(layers.ZeroPadding2D(padding))
            self.conv_model.add(
                layers.MaxPooling2D(pooling_kernel_size, pooling_stride, "same")
            )

        self.positional_emb = positional_emb

    def call(self, images):
        outputs = self.conv_model(images)
        # After passing the images through our mini-network the spatial dimensions
        # are flattened to form sequences.
        reshaped = tf.reshape(
            outputs,
            (
                -1,
                tf.shape(outputs)[1] * tf.shape(outputs)[2],
                tf.shape(outputs)[-1],
            ),
        )
        return reshaped


class PositionEmbedding(keras.layers.Layer):
    def __init__(
        self,
        sequence_length,
        initializer="glorot_uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sequence_length is None:
            raise ValueError("`sequence_length` must be an Integer, received `None`.")
        self.sequence_length = int(sequence_length)
        self.initializer = keras.initializers.get(initializer)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "initializer": keras.initializers.serialize(self.initializer),
            }
        )
        return config

    def build(self, input_shape):
        feature_size = input_shape[-1]
        self.position_embeddings = self.add_weight(
            name="embeddings",
            shape=[self.sequence_length, feature_size],
            initializer=self.initializer,
            trainable=True,
        )

        super().build(input_shape)

    def call(self, inputs, start_index=0):
        inputs_shape = tf.shape(inputs)
        feature_length = inputs_shape[-1]
        sequence_length = inputs_shape[-2]

        position_embeddings = tf.convert_to_tensor(self.position_embeddings)
        position_embeddings = tf.slice(
            position_embeddings,
            (start_index, 0),
            (sequence_length, feature_length),
        )
        return tf.broadcast_to(position_embeddings, inputs_shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class SequencePooling(layers.Layer):
    def __init__(self):
        super().__init__()
        self.attention = layers.Dense(1)

    def call(self, x):
        attention_weights = tf.nn.softmax(self.attention(x), axis=1)
        attention_weights = tf.transpose(attention_weights, perm=(0, 2, 1))
        weighted_representation = tf.matmul(attention_weights, x)
        return tf.squeeze(weighted_representation, -2)


class StochasticDepth(layers.Layer):
    def __init__(self, drop_prop, **kwargs):
        super().__init__(**kwargs)
        self.drop_prob = drop_prop
        self.seed_generator = tf.random.set_seed(1337)

    def call(self, x, training=None):
        if training:
            keep_prob = 1 - self.drop_prob
            shape = (np.shape(x)[0],) + (1,) * (len(x.shape) - 1)
            random_tensor = keep_prob + tf.random.uniform(
                shape, 0, 1, seed=self.seed_generator
            )
            random_tensor = tf.math.floor(random_tensor)
            return (x / keep_prob) * random_tensor
        return x


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation="elu")(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


projection_dim = 64

transformer_units = [
    projection_dim,
    projection_dim,
]


def CCT(
    input_shape,
    num_heads=2,
    projection_dim=projection_dim,
    transformer_units=transformer_units,
):
    inputs = layers.Input(input_shape)

    # Encode patches.
    cct_tokenizer = CCTTokenizer()
    encoded_patches = cct_tokenizer(inputs)

    # Apply positional embedding.
    if positional_emb:
        sequence_length = encoded_patches.shape[1]
        encoded_patches += PositionEmbedding(sequence_length=sequence_length)(
            encoded_patches
        )

    # Calculate Stochastic Depth probabilities.
    dpr = [x for x in np.linspace(0, stochastic_depth_rate, transformer_layers)]

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-5)(x2)

        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

    # Classify outputs.
    x = Flatten()(representation)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=x)
    return model
