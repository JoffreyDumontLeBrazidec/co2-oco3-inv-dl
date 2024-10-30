# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import numpy as np
import tensorflow as tf
from icecream import ic

import data.utils as data_utils
from models.preprocessing import (CloudsLayer, ConditionalNoiseLayer,
                                  TrainingTimeNormalization)


@dataclass
class SegmentationGenerator:
    """Generator with option to augment both images and labels."""

    model_purpose: str
    batch_size: int = 32
    rotation_range: int = 0
    shift_range: float = 0
    flip: bool = False
    shear_range: float = 0
    zoom_range: float = 0
    shuffle: bool = False

    def __post_init__(self):
        self.createDataGenerator()

    def createDataGenerator(self):
        """Create data generator."""

        data_gen_args = dict(
            rotation_range=self.rotation_range,
            width_shift_range=self.shift_range,
            height_shift_range=self.shift_range,
            horizontal_flip=self.flip,
            vertical_flip=self.flip,
            shear_range=self.shear_range,
            zoom_range=self.zoom_range,
        )

        self.image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            **data_gen_args
        )

        if self.model_purpose.startswith("segmentation"):
            self.mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                **data_gen_args
            )

    def flow(self, x_data, y_data):
        """
        Flow on x (img) and y (label) data to generate:
        - segmentation: augmented images and augmented corresponding labels
        - regression: augmented images and non-augmented corresponding labels
        (emissions rate kept unchanged).
        """

        seed = 27

        if self.model_purpose.startswith("segmentation"):
            self.image_generator = self.image_datagen.flow(
                x_data, seed=seed, batch_size=self.batch_size, shuffle=self.shuffle
            )
            self.mask_generator = self.mask_datagen.flow(
                y_data, seed=seed, batch_size=self.batch_size, shuffle=self.shuffle
            )

            self.train_generator = zip(self.image_generator, self.mask_generator)

        elif self.model_purpose == "inversion":
            self.train_generator = self.image_datagen.flow(
                x_data,
                y_data,
                seed=seed,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
            )

        else:
            print("Unknown model purpose in Generator")

        return self.train_generator

    def next(self):
        return self.image_generator.next(), self.mask_generator.next()


@dataclass
class ScaleDataGen(tf.keras.utils.Sequence):
    """
    Custom generator to produce pairs of background+s*plume and s*emissions for inversion.
    s is uniform random and generated between 0.5 and 3.
    """

    x: np.ndarray
    xco2_plume: np.ndarray
    xco2_back: np.ndarray
    xco2_alt_anthro: np.ndarray
    no2_plume: np.ndarray
    no2_back: np.ndarray
    y: np.ndarray
    xco2_chan: list
    no2_chan: list
    input_size: tuple
    batch_size: int = 32
    shuffle: bool = True
    xco2_plume_scaling_min: float = 0.25
    xco2_plume_scaling_max: float = 2
    no2_plume_scaling_min: float = 0.5
    no2_plume_scaling_max: float = 2
    beta_a: float = -30
    beta_b: float = 85

    def __post_init__(self):
        self.N_data = self.x.shape[0]
        self.plume_ids = np.arange(self.N_data)
        self.back_ids = np.arange(self.N_data)
        self.xco2_alt_ids = np.arange(self.N_data)

        self.no2_plume_ids = np.arange(self.N_data)
        self.no2_back_ids = np.arange(self.N_data)

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.plume_ids)
            np.random.shuffle(self.back_ids)
            np.random.shuffle(self.xco2_alt_ids)

    def __get_input(
        self,
        plume_batches: list,
        back_batches: list,
        xco2_alt_batches: list,
        xco2_plume_scaling: np.ndarray,
        back_scaling: np.ndarray,
        xco2_alt_scaling: np.ndarray,
        no2_plume_scaling: np.ndarray,
    ):
        """Get input batches with random scaling."""
        # x_batch = np.empty(shape=(self.batch_size,) + tuple(self.input_size))
        x_batch = self.x[plume_batches]

        for idx, chan in enumerate(self.xco2_chan):
            if chan:
                x_batch[:, :, :, idx : idx + 1] += (
                    xco2_plume_scaling.reshape(xco2_plume_scaling.shape + (1,) * 3)
                    * self.xco2_plume[idx][plume_batches]
                )

                x_batch[:, :, :, idx : idx + 1] += (
                    back_scaling.reshape(back_scaling.shape + (1,) * 3)
                    - self.xco2_back[idx][plume_batches]
                    + self.xco2_back[idx][back_batches]
                )

                x_batch[:, :, :, idx : idx + 1] += (
                    xco2_alt_scaling.reshape(xco2_alt_scaling.shape + (1,) * 3)
                    * self.xco2_alt_anthro[idx][xco2_alt_batches]
                    - self.xco2_alt_anthro[idx][plume_batches]
                )

        for idx, chan in enumerate(self.no2_chan):
            if chan:
                x_batch[:, :, :, idx : idx + 1] += (
                    no2_plume_scaling.reshape(no2_plume_scaling.shape + (1,) * 3)
                    * self.no2_plume[idx][plume_batches]
                )

                x_batch[:, :, :, idx : idx + 1] += (
                    back_scaling.reshape(back_scaling.shape + (1,) * 3)
                    - self.no2_back[idx][plume_batches]
                    + self.no2_back[idx][back_batches]
                )

        return x_batch

    def __get_output(self, batches: list, plume_scaling: np.ndarray):
        """Get output batches with random scaling."""
        y_batch = (
            self.y[batches]
            + plume_scaling.reshape(plume_scaling.shape + (1,) * 1) * self.y[batches]
        )
        return y_batch

    def __get_data(
        self, plume_batches: list, back_batches: list, xco2_alt_batches: list
    ):
        """Get random batches, drawing random scaling."""
        xco2_plume_scaling = np.random.uniform(
            self.xco2_plume_scaling_min - 1,
            self.xco2_plume_scaling_max,
            size=self.batch_size,
        )
        back_scaling = np.random.uniform(-3.5, 3.5, size=self.batch_size)
        xco2_alt_scaling = np.random.uniform(0.33, 3, size=self.batch_size)

        no2_plume_scaling = np.random.uniform(
            self.no2_plume_scaling_min - 1,
            self.no2_plume_scaling_max,
            size=self.batch_size,
        )

        x_batch = self.__get_input(
            plume_batches,
            back_batches,
            xco2_alt_batches,
            xco2_plume_scaling,
            back_scaling,
            xco2_alt_scaling,
            no2_plume_scaling,
        )
        y_batch = self.__get_output(plume_batches, xco2_plume_scaling)
        return x_batch, y_batch

    def __getitem__(self, index: int):
        """Get random list of batches to draw data."""
        plume_batches = self.plume_ids[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]
        back_batches = self.back_ids[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]  # type: ignore
        xco2_alt_batches = self.xco2_alt_ids[
            range(index * self.batch_size, (index + 1) * self.batch_size)
        ]  # type: ignore
        x, y = self.__get_data(plume_batches, back_batches, xco2_alt_batches)
        return x, y

    def __len__(self):
        """Get number of batches per epoch."""
        return self.N_data // self.batch_size


def generate_beta_samples(
    shape: list,
    a: float = -30,
    b: float = 85,
    alpha: int = 8,
    beta: int = 10,
    lower_bound: float = 0,
    upper_bound: float = 45,
):
    """Generate samples from beta distribution."""
    total_samples = shape[0]
    estimated_samples_needed = int(total_samples * 1.3)
    final_samples = np.empty(shape=shape[0])
    N_valid_samples_prec = 0

    while True:
        batch_samples = np.random.beta(alpha, beta, size=estimated_samples_needed)
        scaled_batch_samples = a + (b - a) * batch_samples

        # Filter samples within the desired range
        valid_samples = scaled_batch_samples[
            (scaled_batch_samples >= lower_bound)
            & (scaled_batch_samples <= upper_bound)
        ]
        N_valid_samples = valid_samples.shape[0]
        N_samples_to_feed = np.min(
            (N_valid_samples, total_samples - N_valid_samples_prec)
        )
        final_samples[
            N_valid_samples_prec : N_samples_to_feed + N_valid_samples_prec
        ] = valid_samples[:N_samples_to_feed]
        N_valid_samples_prec = N_valid_samples

        if N_valid_samples >= total_samples:
            return final_samples.reshape(shape)
        else:
            # Increase the estimated number of samples needed for the next iteration
            estimated_samples_needed = int((total_samples - N_samples_to_feed) * 1.5)


@dataclass
class InvDataGen(tf.keras.utils.Sequence):
    """
    Custom generator to
    - produce pairs of background+s*plume and s*emissions for inversion.
    - noise data
    - normalise data
    - add clouds to data
    """

    x: Any
    xco2_plume: np.ndarray
    xco2_back: np.ndarray
    xco2_alt_anthro: np.ndarray
    no2_plume: Optional[np.ndarray]
    no2_back: Optional[np.ndarray]
    y: np.ndarray
    list_chans: list
    clouds: np.ndarray
    noise_layer: ConditionalNoiseLayer
    normalise_layer: TrainingTimeNormalization
    cloud_layer: CloudsLayer
    batch_size: int = 128
    shuffle: bool = True
    xco2_plume_scaling_min: float = 0.33
    xco2_plume_scaling_max: float = 2.25
    no2_plume_scaling_min: float = 0.75
    no2_plume_scaling_max: float = 2
    xco2_emiss_min: float = 0
    xco2_emiss_max: float = 45
    scaling_method: str = "beta_distribution_mapping"
    beta_a: float = -30
    beta_b: float = 85

    # can also be "uniform_emiss"
    def __post_init__(self):
        self.N_data = self.xco2_plume.shape[0]

        self.plume_ids = np.arange(self.N_data)
        self.back_ids = np.arange(self.N_data)
        self.xco2_alt_ids = np.arange(self.N_data)
        self.clouds_ids = np.arange(self.clouds.shape[0])

    def on_epoch_end(self):
        """Shuffle data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.plume_ids)
            np.random.shuffle(self.back_ids)
            np.random.shuffle(self.xco2_alt_ids)
            np.random.shuffle(self.clouds_ids)

    def __get_input(
        self,
        plume_batches: np.ndarray,
        back_batches: np.ndarray,
        xco2_alt_batches: np.ndarray,
        cloud_batches: np.ndarray,
        xco2_plume_scaling: np.ndarray,
        back_scaling: np.ndarray,
        xco2_alt_scaling: np.ndarray,
        no2_plume_scaling: np.ndarray,
    ):
        """Get input batches with random scaling."""
        if isinstance(self.x, np.ndarray):
            x_batch = self.x[plume_batches]
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            x_batch = self.x[0][plume_batches]
        else:
            ic("Error in generator")
            sys.exit()

        for idx, chan in enumerate(self.list_chans):
            if chan == "xco2":
                x_batch[:, :, :, idx : idx + 1] = (
                    (
                        xco2_plume_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        * self.xco2_plume[plume_batches]
                    )
                    + (
                        back_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        + self.xco2_back[back_batches]
                    )
                    + (
                        xco2_alt_scaling[:, np.newaxis, np.newaxis, np.newaxis]
                        * self.xco2_alt_anthro[xco2_alt_batches]
                    )
                )
            elif chan == "no2":
                x_batch[:, :, :, idx : idx + 1] = (
                    no2_plume_scaling.reshape(no2_plume_scaling.shape + (1,) * 3)
                    * self.no2_plume[plume_batches]
                ) + self.no2_back[back_batches]

        x_batch = self.noise_layer(x_batch, training=True)
        x_batch = self.normalise_layer(x_batch, training=True)
        x_batch = self.cloud_layer.apply_clouds_to_field(
            x_batch, self.clouds[cloud_batches]
        )

        return x_batch

    def __get_output(self, batches: np.ndarray, plume_scaling: np.ndarray):
        """Get output batches with random scaling."""
        y_batch = plume_scaling[:, np.newaxis] * self.y[batches]

        return y_batch

    def __calculate_xco2_plume_scaling(self, plume_batches: np.ndarray):
        """Calculate xco2 plume scaling within specified bounds."""
        if self.scaling_method == "uniform_scaling":
            y_values = self.y[plume_batches]

            scaling_min = np.maximum(
                self.xco2_emiss_min / y_values, self.xco2_plume_scaling_min
            )
            scaling_max = np.minimum(
                self.xco2_emiss_max / y_values, self.xco2_plume_scaling_max
            )
            return np.squeeze(np.random.uniform(scaling_min, scaling_max))

        elif self.scaling_method == "uniform_distribution_mapping":
            y_values = self.y[plume_batches]
            random_uniform_value = np.random.uniform(
                self.xco2_emiss_min, self.xco2_emiss_max, size=y_values.shape
            )
            scaling = random_uniform_value / y_values

            return np.squeeze(scaling)

        elif self.scaling_method == "beta_distribution_mapping":
            y_values = self.y[plume_batches]
            scaled_samples = generate_beta_samples(
                y_values.shape,
                lower_bound=self.xco2_emiss_min,
                upper_bound=self.xco2_emiss_max,
                a=self.beta_a,
                b=self.beta_b,
            )
            scaling = scaled_samples / y_values
            return np.squeeze(scaling)
        else:
            ic("Typo in scaling method decleared.")
            sys.exit()

    def __get_data(
        self,
        plume_batches: np.ndarray,
        back_batches: np.ndarray,
        xco2_alt_batches: np.ndarray,
        cloud_batches: np.ndarray,
    ):
        """Get random batches, drawing random scaling."""

        xco2_plume_scaling = self.__calculate_xco2_plume_scaling(plume_batches)

        back_scaling = np.random.uniform(-3.5, 3.5, size=self.batch_size)
        xco2_alt_scaling = np.random.uniform(0.25, 3, size=self.batch_size)
        no2_plume_scaling = np.random.uniform(
            self.no2_plume_scaling_min,
            self.no2_plume_scaling_max,
            size=self.batch_size,
        )
        x_batch = self.__get_input(
            plume_batches,
            back_batches,
            xco2_alt_batches,
            cloud_batches,
            xco2_plume_scaling,
            back_scaling,
            xco2_alt_scaling,
            no2_plume_scaling,
        )

        y_batch = self.__get_output(plume_batches, xco2_plume_scaling)

        if isinstance(self.x, np.ndarray):
            return x_batch, y_batch
        elif isinstance(self.x, list) and all(
            isinstance(arr, np.ndarray) for arr in self.x
        ):
            return [x_batch, self.x[1][plume_batches]], y_batch
        else:
            sys.exit()

    def __getitem__(self, index: int):
        """Get random list of batches to draw data."""
        plume_batches = self.plume_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        back_batches = self.back_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        xco2_alt_batches = self.xco2_alt_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        cloud_batches = self.clouds_ids[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        x, y = self.__get_data(
            plume_batches, back_batches, xco2_alt_batches, cloud_batches
        )
        return x, y

    def __len__(self):
        """Get number of batches per epoch."""
        return self.N_data // self.batch_size


@dataclass
class InvUncertaintyGen(InvDataGen):
    """Generator used to train the uncertainty quantification model."""

    path_inv_model: Optional[str] = None
    metric: tf.keras.losses.MeanAbsoluteError = tf.keras.losses.MeanAbsoluteError(
        reduction=tf.keras.losses.Reduction.NONE
    )

    def __post_init__(self):
        super().__post_init__()

        if self.path_inv_model:
            self.inv_model = data_utils.get_inversion_model(
                self.path_inv_model,
                name_w="w_last.h5",
            )

        else:
            logging.error("Missing path_inv_model in generators.py")
            sys.exit()

    def __getitem__(self, index):
        # Call the superclass method to perform standard operations

        x_batch, y_batch = super().__getitem__(index)
        if self.inv_model:
            inv_model_estimations = self.metric(
                y_batch,
                self.inv_model.predict(
                    x_batch, batch_size=self.batch_size, verbose=None
                ),
            )
            y_batch = tf.reshape(inv_model_estimations, y_batch.shape)
        else:
            sys.exit()

        return x_batch, y_batch
