# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import joblib
import numpy as np
import tensorflow as tf
import xarray as xr
from icecream import ic
from sklearn import preprocessing

import data.utils as data_utils
from models.preprocessing import (
    CloudsLayer,
    ConditionalNoiseLayer,
    TrainingTimeNormalization,
)


@dataclass
class Input_filler:
    """Fill chans for Input_train and Input_eval."""

    dir_seg_models: str = "None"
    noise_level: float = 0.7
    noise: bool = True

    def fill_data(self, ds: xr.Dataset, list_chans: list) -> np.ndarray:
        """Fill input data according to chan_0,1,2"""
        data = self.fill_chan(list_chans[0], ds)
        for chan in [x for x in list_chans[1:] if x != "None"]:
            data = np.concatenate((data, self.fill_chan(chan, ds)), axis=-1)

        return data

    def fill_chan(self, chan: str, ds: xr.Dataset) -> np.ndarray:
        """Return array depending on chan type specified."""
        if chan == "xco2":
            data_chan = data_utils.get_xco2_noisy(ds)
        elif chan == "xco2_train":
            data_chan = data_utils.get_xco2_noiseless(ds)
        elif chan == "u_wind" or chan == "u_wind_train":
            data_chan = data_utils.get_u_wind(ds)
        elif chan == "v_wind" or chan == "v_wind_train":
            data_chan = data_utils.get_v_wind(ds)
        elif chan == "no2":
            data_chan = data_utils.get_no2_noiseless(ds)
        elif chan == "no2_train":
            data_chan = data_utils.get_no2_noisy(ds)
        elif chan == "seg_pred_no2":
            data_chan = data_utils.get_seg_pred_no2(ds)

        elif chan == "plume":
            data_chan = data_utils.get_xco2_plume(ds)
        elif chan == "bool_perf_seg":
            data_chan = data_utils.get_bool_perf_seg(ds)
        elif chan == "weighted_plume":
            data_chan = data_utils.get_weighted_plume(ds)
        else:
            logging.error("Issue with Input_filler.fill_chan")
            sys.exit()

        return data_chan


@dataclass
class Input_train:
    """Prepare and store train and valid inputs."""

    ds_train: xr.Dataset
    ds_valid: xr.Dataset
    ds_valid_2: xr.Dataset
    ds_extra_valid: xr.Dataset
    chan_0: str
    chan_1: str
    chan_2: str
    chan_3: str
    chan_4: str
    clouds_threshold: float
    dir_clouds_array: str
    dir_seg_models: str
    timedate: bool

    noise_level: float = 0.7

    def __post_init__(self):
        self.make_list_chans()
        self.fill_data()

        self.prepare_fields_generation()

        self.get_noise_layer()
        self.get_norm_layer()
        self.normalise_validation_data()
        self.get_cloud_layer()
        self.cloud_validation_data()

        self.add_timedate_vector()

    def make_list_chans(self):
        """Make list of channels."""
        self.list_chans = [
            self.chan_0,
            self.chan_1,
            self.chan_2,
            self.chan_3,
            self.chan_4,
        ]
        while self.list_chans[-1] == "None":
            del self.list_chans[-1]

        """Append '_train' to all channel names in the list of channels."""
        self.train_list_chans = [chan + "_train" for chan in self.list_chans]

    def get_noise_layer(self):
        """
        Evaluate channels with noise:
        - training channels should not be noisy (they will be noised during training)
        - valid channels are noisy
        """
        self.noise_layer = ConditionalNoiseLayer(self.list_chans)

    def fill_data(self):
        """Fill data.x.train with channels choice."""
        filler = Input_filler(
            self.dir_seg_models,
            self.noise_level,
        )
        self.train = filler.fill_data(self.ds_train, self.train_list_chans)
        self.valid = filler.fill_data(self.ds_valid, self.list_chans)
        self.valid_2 = filler.fill_data(self.ds_valid_2, self.list_chans)
        self.extra_valid = filler.fill_data(self.ds_extra_valid, self.list_chans)

        self.fields_input_shape = list(self.train.shape[1:])

    def get_norm_layer(self):
        """Get normalisation layer and adapt it to data.x.train."""
        self.norm_layer = TrainingTimeNormalization(axis=-1, name="norm_layer")
        self.norm_layer.adapt(self.train)

    def normalise_validation_data(self):
        """Normalise validation data with adapted n_layer."""
        self.valid = np.array(self.norm_layer(self.valid, training=True))
        self.valid_2 = np.array(self.norm_layer(self.valid_2, training=True))
        self.extra_valid = np.array(self.norm_layer(self.extra_valid, training=True))

    def get_cloud_layer(self):
        """Get cloud layer and adapt it to data.x.train"""
        self.clouds_training = data_utils.get_clouds_array(
            self.dir_clouds_array,
            "train",
            self.clouds_threshold,
            self.fields_input_shape[1],
        )
        self.cloud_layer = CloudsLayer(
            self.clouds_training, self.list_chans, name="cloud_layer"
        )
        self.cloud_layer.evaluate_nanvalues(self.norm_layer(self.train, training=True))

    def cloud_validation_data(self):
        """Add clouds on validation, validation_2, extra_validation data."""

        fields = ["valid", "valid_2", "extra_valid"]
        for field in fields:
            setattr(
                self,
                field,
                self.cloud_layer.apply_clouds_to_field(
                    getattr(self, field),
                    data_utils.get_clouds_array(
                        self.dir_clouds_array,
                        field,
                        self.clouds_threshold,
                        self.fields_input_shape[1],
                    )[: getattr(self, field).shape[0]],
                ),
            )

    def prepare_fields_generation(self):
        """Prepare for scaling. Get plume in independant array and boolean channels."""
        for _, chan in enumerate(self.list_chans):
            if chan == "xco2":
                self.xco2_plumes_train = data_utils.get_xco2_plume(self.ds_train)
                self.xco2_back_train = data_utils.get_xco2_back(self.ds_train)
                self.xco2_alt_anthro_train = data_utils.get_xco2_alt_anthro(
                    self.ds_train
                )
            elif chan == "no2":
                self.no2_plumes_train = data_utils.get_no2_plume(self.ds_train)
                self.no2_back_train = data_utils.get_no2_noiseless(
                    self.ds_train
                ) - data_utils.get_no2_plume(self.ds_train)

    def add_timedate_vector(self):
        """Get timedate vector."""
        self.timedate_vector_size = 0
        if self.timedate:
            timedate_vector_train = data_utils.get_timedate_vector(self.ds_train)
            self.timedate_vector_size = timedate_vector_train.shape[1]
            scaler = preprocessing.MinMaxScaler()
            norm_timedate_vector = scaler.fit_transform(timedate_vector_train)
            self.train = [self.train, norm_timedate_vector]
            # self.timedate_train = norm_timedate_vector

            fields = ["valid", "valid_2", "extra_valid"]
            for field in fields:
                timedate_vector_valid = data_utils.get_timedate_vector(
                    getattr(self, f"ds_{field}")
                )
                norm_timedate_vector_valid = scaler.transform(timedate_vector_valid)
                current_field_value = getattr(self, field)
                setattr(self, field, [current_field_value, norm_timedate_vector_valid])
                # setattr(self, f"timedate_{field}", norm_timedate_vector_valid)


@dataclass
class Output_train:
    """Prepare and store train and valid outputs."""

    ds_train: xr.Dataset
    ds_valid: xr.Dataset
    ds_valid_2: xr.Dataset
    ds_extra_valid: xr.Dataset
    classes: int

    def get_segmentation(self, curve, min_w, max_w, param_curve):
        """Get segmentation train and valid."""
        self.train = data_utils.get_weighted_plume(
            self.ds_train, curve, min_w, max_w, param_curve
        )
        self.valid = data_utils.get_weighted_plume(
            self.ds_valid, curve, min_w, max_w, param_curve
        )
        self.extra_valid = data_utils.get_weighted_plume(
            self.ds_extra_valid, curve, min_w, max_w, param_curve
        )
        print("data.y.train.shape", self.train.shape)

    def get_inversion(self, N_hours_prec):
        """Get inversion train and valid."""
        self.train = data_utils.get_emiss(self.ds_train, N_hours_prec)
        self.valid = data_utils.get_emiss(self.ds_valid, N_hours_prec)
        self.valid_2 = data_utils.get_emiss(self.ds_valid_2, N_hours_prec)
        self.extra_valid = data_utils.get_emiss(self.ds_extra_valid, N_hours_prec)


@dataclass
class Data_train:
    """Object for containing Input and Output data and all other informations."""

    path_train_ds: str
    path_valid_ds: str
    path_valid_2_ds: str = "None"
    path_extra_valid_ds: str = "None"

    def __post_init__(self):
        self.ds_train = xr.open_dataset(self.path_train_ds)
        self.ds_valid = xr.open_dataset(self.path_valid_ds)

        try:
            self.ds_valid_2 = xr.open_dataset(self.path_valid_2_ds)
        except:
            self.ds_valid_2 = self.ds_valid

        try:
            self.ds_extra_valid = xr.open_dataset(self.path_extra_valid_ds)
        except:
            self.ds_extra_valid = self.ds_valid

    def prepare_input(
        self,
        chan_0: str,
        chan_1: str = "None",
        chan_2: str = "None",
        chan_3: str = "None",
        chan_4: str = "None",
        clouds_threshold: float = 0.5,
        dir_clouds_array: str = "/libre/dumontj/coco2/dl-input/clouds",
        dir_seg_models: str = "/cerea_raid/users/dumontj/dev/coco2/dl/res/models",
        timedate: bool = False,
    ):
        """Prepare input object."""
        self.x = Input_train(
            self.ds_train,
            self.ds_valid,
            self.ds_valid_2,
            self.ds_extra_valid,
            chan_0,
            chan_1,
            chan_2,
            chan_3,
            chan_4,
            clouds_threshold=clouds_threshold,
            dir_clouds_array=dir_clouds_array,
            dir_seg_models=dir_seg_models,
            timedate=timedate,
        )

    def prepare_output_segmentation(
        self,
        curve: str = "linear",
        min_w: float = 0.01,
        max_w: float = 4,
        param_curve: float = 1,
    ):
        """Prepare output object for segmentation."""
        self.y = Output_train(
            self.ds_train,
            self.ds_valid,
            self.ds_valid_2,
            self.ds_extra_valid,
            classes=1,
        )
        self.y.get_segmentation(curve, min_w, max_w, param_curve)

    def prepare_output_inversion(self, N_hours_prec: int = 1):
        """Prepare output object for inversion."""
        self.y = Output_train(
            self.ds_train,
            self.ds_valid,
            self.ds_valid_2,
            self.ds_extra_valid,
            classes=1,
        )
        self.y.get_inversion(N_hours_prec=N_hours_prec)


@dataclass
class Input_eval:
    """Prepare and store train and valid inputs."""

    ds_eval: xr.Dataset
    chan_0: str
    chan_1: str
    chan_2: str
    chan_3: str
    chan_4: str
    clouds_threshold: float
    dir_clouds_array: str
    norm_model: tf.keras.models.Model
    cloud_model: Optional[tf.keras.models.Model] = None
    dir_seg_models: str = "None"
    noise_level: float = 0.7
    timedate: bool = False

    def __post_init__(self):
        self.make_list_chans()
        self.fill_data()

        self.normalise_data()
        self.cloud_data()

    def make_list_chans(self):
        """Make list of channels."""
        self.list_chans = [
            self.chan_0,
            self.chan_1,
            self.chan_2,
            self.chan_3,
            self.chan_4,
        ]
        while self.list_chans[-1] == "None":
            del self.list_chans[-1]

    def fill_data(self):
        """Fill data.x.eval with channels choice"""
        filler = Input_filler(
            self.dir_seg_models,
            self.noise_level,
        )

        self.eval = filler.fill_data(self.ds_eval, self.list_chans)
        ic(self.eval.shape)

        self.fields_input_shape = list(self.eval.shape[1:])

    def normalise_data(self):
        """Normalise data with adapted n_layer."""
        self.eval = np.array(self.norm_model(self.eval, training=True))

    def cloud_data(self):
        """Add clouds on data."""
        if self.cloud_model:
            self.eval = self.cloud_model.get_layer("cloud_layer").apply_clouds_to_field(
                self.eval,
                data_utils.get_clouds_array(
                    self.dir_clouds_array,
                    "eval",
                    self.clouds_threshold,
                    self.fields_input_shape[1],
                )[: self.eval.shape[0]],
            )

    def add_timedate_vector(self):
        """Add timedate vector to data.x.eval."""
        self.timedate_vector_size = 0
        if self.timedate:
            timedate_vector = data_utils.get_timedate_vector(self.ds_eval)
            scaler = preprocessing.MinMaxScaler()
            norm_timedate_vector = scaler.fit_transform(timedate_vector)
            # note: not very valid, but the scaler actually does no transform the data that are already
            # between 0 and 1
            self.eval = [self.eval, norm_timedate_vector]


@dataclass
class Output_eval:
    """Prepare and store train and valid outputs."""

    ds_eval: xr.Dataset
    classes: int

    def get_segmentation(self, curve, min_w, max_w, param_curve):
        """Get segmentation train and valid."""
        self.eval = data_utils.get_weighted_plume(
            self.ds_eval, curve, min_w, max_w, param_curve
        )
        print("data.y.train.shape", self.eval.shape)

    def get_inversion(self, N_hours_prec):
        """Get inversion train and valid."""
        self.eval = data_utils.get_emiss(self.ds_eval, N_hours_prec)


@dataclass
class Data_eval:
    path_eval_nc: str

    def __post_init__(self):
        self.ds = xr.open_dataset(self.path_eval_nc)

    def prepare_input(
        self,
        chan_0: str,
        chan_1: str = "None",
        chan_2: str = "None",
        chan_3: str = "None",
        chan_4: str = "None",
        clouds_threshold: float = 0.5,
        dir_clouds_array: str = "/libre/dumontj/coco2/dl-input/clouds",
        norm_model: Optional[tf.keras.models.Model] = None,
        cloud_model: Optional[tf.keras.models.Model] = None,
        dir_seg_models: str = "/cerea_raid/users/dumontj/dev/coco2/dl/res/models",
        timedate: bool = False,
    ):
        """Prepare input object."""
        self.x = Input_eval(
            self.ds,
            chan_0,
            chan_1,
            chan_2,
            chan_3,
            chan_4,
            clouds_threshold,
            dir_clouds_array,
            norm_model,
            cloud_model,
            dir_seg_models=dir_seg_models,
            timedate=timedate,
        )

    def prepare_output_segmentation(
        self,
        curve: str = "linear",
        min_w: float = 0.01,
        max_w: float = 4,
        param_curve: float = 1,
    ):
        """Prepare output object for segmentation."""
        self.y = Output_eval(self.ds, classes=1)
        self.y.get_segmentation(curve, min_w, max_w, param_curve)

    def prepare_output_inversion(self, N_hours_prec: int = 1):
        """Prepare output object for inversion."""
        self.y = Output_eval(self.ds, classes=1)
        self.y.get_inversion(N_hours_prec=N_hours_prec)
