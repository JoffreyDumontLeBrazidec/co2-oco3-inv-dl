# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2023
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import math
import os
import sys
from dataclasses import dataclass, field

import joblib
import numpy as np
import tensorflow as tf
import xarray as xr
from sklearn import preprocessing

from include.loss import calculate_weighted_plume, pixel_weighted_cross_entropy


def get_scaler(dir_res: str) -> preprocessing.StandardScaler:
    """Get scaler (fit on training data) for evaluation."""
    scaler = joblib.load(os.path.join(dir_res, "scaler.save"))
    return scaler


def get_xco2_noisy(ds: xr.Dataset, noise_level: float = 0.7):
    """Return noisy xco2 field related to ds."""
    xco2 = np.expand_dims(ds.xco2.values, -1)
    noise = noise_level * np.random.randn(*xco2.shape)
    xco2 = xco2 + noise
    return xco2


def get_xco2_noisy_prec(ds: xr.Dataset, noise_level: float = 0.7):
    """Return noisy xco2 field related to ds at the previous time."""
    xco2 = np.concatenate((ds.xco2.values[0:1], ds.xco2.values[0:-1]))
    xco2 = np.expand_dims(ds.xco2.values, -1)
    noise = noise_level * np.random.randn(*xco2.shape)
    xco2 = xco2 + noise
    return xco2


def get_xco2_noiseless(ds: xr.Dataset):
    """Return noiseless xco2 field related to ds."""
    return np.expand_dims(ds.xco2.values, -1)


def get_xco2_noiseless_prec(ds: xr.Dataset, noise_level: float = 0.7):
    """Return noiseless xco2 field related to ds at the previous time."""
    xco2 = np.concatenate((ds.xco2.values[0:1], ds.xco2.values[0:-1]))
    return np.expand_dims(xco2, -1)


def get_no2_noisy(ds: xr.Dataset):
    """Return noisy no2 field related to ds."""
    no2 = np.expand_dims(ds.no2.values, -1)
    no2_noisy = no2 + np.random.randn(*no2.shape) * no2
    return no2_noisy


def get_no2_noiseless(ds: xr.Dataset):
    """Return noiseless no2 field related to ds."""
    return np.expand_dims(ds.no2.values, -1)


def get_u_wind(ds: xr.Dataset) -> np.ndarray:
    """Return u wind related to ds."""
    return np.expand_dims(ds.u.values, -1)


def get_plume(ds: xr.Dataset) -> np.ndarray:
    """Return plume from ds."""
    return np.expand_dims(ds.plume.values, -1)


def get_plume_prec(ds: xr.Dataset) -> np.ndarray:
    """Return plume from ds at the previous time."""
    plume = np.concatenate((ds.plume.values[0:1], ds.plume.values[0:-1]))
    return np.expand_dims(plume, -1)


def get_bool_perf_seg(ds: xr.Dataset) -> np.ndarray:
    """Return boolean perfect segmentations from ds."""
    return np.expand_dims(ds.bool_perf_seg.values, -1)


def get_v_wind(ds: xr.Dataset) -> np.ndarray:
    """Return u wind related to ds."""
    return np.expand_dims(ds.v.values, -1)


def get_emiss(ds: xr.Dataset, N_hours_prec: int) -> np.ndarray:
    """Return emiss array related to ds."""
    emiss = np.array(ds.emiss.values, dtype=float)
    emiss = emiss[:, 1 : N_hours_prec + 1]
    return emiss


def get_bool_(ds: xr.Dataset, N_hours_prec: int) -> np.ndarray:
    """Return emiss array related to ds."""
    emiss = np.array(ds.emiss.values, dtype=float)
    emiss = emiss[:, 1 : N_hours_prec + 1]
    return emiss


def get_weighted_plume(
    ds: xr.Dataset,
    curve: str = "linear",
    min_w: float = 0,
    max_w: float = 1,
    param_curve: float = 1,
):
    """Get modified plume matrices label output."""
    y_data = calculate_weighted_plume(
        np.array(ds.plume.values, dtype=float), min_w, max_w, curve, param_curve
    )
    return y_data


def get_weighted_plume_prec(
    ds: xr.Dataset,
    curve: str = "linear",
    min_w: float = 0,
    max_w: float = 1,
    param_curve: float = 1,
):
    """Get modified plume matrices label output."""
    plume = np.concatenate((ds.plume.values[0:1], ds.plume.values[0:-1]))
    y_data = calculate_weighted_plume(
        np.array(plume, dtype=float), min_w, max_w, curve, param_curve
    )
    return y_data


@dataclass
class Segmentations_pred:
    """To return segmentation predictions"""

    dir_seg_models: str
    name_seg_model: str
    ds: xr.Dataset
    model_optimiser: str = "adam"
    model_loss = pixel_weighted_cross_entropy
    noise_level: float = 0.7

    def __post_init__(self):
        self.model = tf.keras.models.load_model(
            os.path.join(
                self.dir_seg_models, f"{self.name_seg_model}", "weights_cp_best.h5"
            ),
            compile=False,
        )
        self.model.compile(self.model_optimiser, loss=self.model_loss)
        self.scaler = get_scaler(os.path.join(self.dir_seg_models, self.name_seg_model))

    def get_seg_predictions(self) -> np.ndarray:
        """Return segmentation predictions from Model on Dataset."""
        xco2_noisy = get_xco2_noisy(self.ds, self.noise_level)
        if self.model.layers[0].input_shape[0][-1] == 1:
            inputs = xco2_noisy
        elif self.model.layers[0].input_shape[0][-1] == 2:
            no2_noisy = get_no2_noisy(self.ds)
            inputs = np.concatenate((xco2_noisy, no2_noisy), axis=-1)
        else:
            sys.exit()

        inputs = np.array(
            self.scaler.transform(inputs.reshape(-1, inputs.shape[-1]))
        ).reshape(inputs.shape)
        inputs = tf.convert_to_tensor(inputs, float)
        x_seg = tf.convert_to_tensor(self.model.predict(inputs), float)
        return np.array(x_seg)


@dataclass
class Input_filler:
    """Fill chans for Input_train and Input_eval."""

    dir_seg_models: str = "None"
    noise_level: float = 0.7

    def fill_data(self, ds: xr.Dataset, list_chans: list) -> np.ndarray:
        """Fill input data according to chan_0,1,2"""
        data = self.fill_chan(list_chans[0], ds)
        for chan in [x for x in list_chans[1:] if x != "None"]:
            data = np.concatenate((data, self.fill_chan(chan, ds)), axis=-1)

        return data

    def fill_chan(self, chan: str, ds: xr.Dataset) -> np.ndarray:
        """Return array depending on chan type specified."""
        if chan == "xco2" or chan == "xco2_noisy":
            data_chan = get_xco2_noisy(ds, self.noise_level)
        elif chan == "xco2_prec" or chan == "xco2_noisy_prec":
            data_chan = get_xco2_noisy_prec(ds, self.noise_level)
        elif chan == "xco2_noiseless":
            data_chan = get_xco2_noiseless(ds)
        elif chan == "xco2_noiseless_prec":
            data_chan = get_xco2_noiseless_prec(ds)
        elif chan == "plume":
            data_chan = get_plume(ds)
        elif chan == "plume_prec":
            data_chan = get_plume_prec(ds)
        elif chan == "bool_perf_seg":
            data_chan = get_bool_perf_seg(ds)
        elif chan == "weighted_plume":
            data_chan = get_weighted_plume(ds)
        elif chan == "weighted_plume_prec":
            data_chan = get_weighted_plume_prec(ds)
        elif chan == "no2":
            data_chan = get_no2_noisy(ds)
        elif chan.startswith("seg"):
            seg_cont = Segmentations_pred(self.dir_seg_models, chan, ds=ds)
            data_chan = seg_cont.get_seg_predictions()
        elif chan == "u_wind":
            data_chan = get_u_wind(ds)
        elif chan == "v_wind":
            data_chan = get_v_wind(ds)
        else:
            sys.exit()

        return data_chan


@dataclass
class Input_train:
    """Prepare and store train and valid inputs."""

    ds_train: xr.Dataset
    ds_valid: xr.Dataset
    ds_extra_valid: xr.Dataset
    chan_0: str
    chan_1: str = "None"
    chan_2: str = "None"
    chan_3: str = "None"
    chan_4: str = "None"
    dir_seg_models: str = "None"
    noise_level: float = 0.7

    def __post_init__(self):

        self.list_chans = [
            self.chan_0,
            self.chan_1,
            self.chan_2,
            self.chan_3,
            self.chan_4,
        ]

        self.evaluate_noise()
        self.fill_data()
        self.get_norm_layer()
        self.prepare_for_scaling()

    def evaluate_noise(self):
        """Evaluate channels with XCO2 noise: replace and memorise."""
        self.xco2_noisy_chans = [False] * len(self.list_chans)
        self.train_list_chans = self.list_chans.copy()
        for idx in range(len(self.list_chans)):
            if self.list_chans[idx] == "xco2" or self.list_chans[idx] == "xco2_noisy":
                self.train_list_chans[idx] = "xco2_noiseless"
                self.xco2_noisy_chans[idx] = True
            elif (
                self.list_chans[idx] == "xco2_prec"
                or self.list_chans[idx] == "xco2_noisy_prec"
            ):
                self.train_list_chans[idx] = "xco2_noiseless_prec"
                self.xco2_noisy_chans[idx] = True
            else:
                pass

    def fill_data(self):
        """Fill data.x.train with channels choice."""
        filler = Input_filler(
            self.dir_seg_models,
            self.noise_level,
        )
        self.train = filler.fill_data(self.ds_train, self.train_list_chans)
        self.valid = filler.fill_data(self.ds_valid, self.list_chans)
        self.extra_valid = filler.fill_data(self.ds_extra_valid, self.list_chans)

        self.fields_input_shape = list(self.train.shape[1:])

    def get_norm_layer(self):
        """Get normalisation layer and adapt it to data.x.train."""
        self.n_layer = tf.keras.layers.Normalization(axis=-1, name="preproc_norm")
        self.n_layer.adapt(self.train)

        print("data.x.train.shape", self.train.shape)

    def prepare_for_scaling(self):
        """Prepare for scaling. Get plume in independant array and boolean channels."""
        self.scale_bool = [False] * len(self.list_chans)
        self.plumes_train = {}
        for idx, chan in enumerate(self.list_chans):
            if chan.startswith("xco2"):
                self.scale_bool[idx] = True
                if "prec" in chan:
                    self.plumes_train[idx] = get_plume_prec(self.ds_train)
                else:
                    self.plumes_train[idx] = get_plume(self.ds_train)


@dataclass
class Output_train:
    """Prepare and store train and valid outputs."""

    ds_train: xr.Dataset
    ds_valid: xr.Dataset
    ds_extra_valid: xr.Dataset
    classes: int

    def get_segmentation(self, curve, min_w, max_w, param_curve):
        """Get segmentation train and valid."""
        self.train = get_weighted_plume(self.ds_train, curve, min_w, max_w, param_curve)
        self.valid = get_weighted_plume(self.ds_valid, curve, min_w, max_w, param_curve)
        self.extra_valid = get_weighted_plume(
            self.ds_extra_valid, curve, min_w, max_w, param_curve
        )
        print("data.y.train.shape", self.train.shape)

    def get_inversion(self, N_hours_prec):
        """Get inversion train and valid."""
        self.train = get_emiss(self.ds_train, N_hours_prec)
        self.valid = get_emiss(self.ds_valid, N_hours_prec)
        self.extra_valid = get_emiss(self.ds_extra_valid, N_hours_prec)


@dataclass
class Data_train:
    """Object for containing Input and Output data and all other informations."""

    path_train_ds: str
    path_valid_ds: str
    path_extra_valid_ds: str = "None"

    def __post_init__(self):

        self.ds_train = xr.open_dataset(self.path_train_ds)
        self.ds_valid = xr.open_dataset(self.path_valid_ds)
        if self.path_extra_valid_ds != "None":
            self.ds_extra_valid = xr.open_dataset(self.path_extra_valid_ds)
        else:
            self.ds_extra_valid = xr.open_dataset(self.path_valid_ds)

    def prepare_input(
        self,
        chan_0: str,
        chan_1: str = "None",
        chan_2: str = "None",
        chan_3: str = "None",
        chan_4: str = "None",
        dir_seg_models: str = "/cerea_raid/users/dumontj/dev/coco2/dl/res/models",
    ):
        """Prepare input object."""
        self.x = Input_train(
            self.ds_train,
            self.ds_valid,
            self.ds_extra_valid,
            chan_0,
            chan_1,
            chan_2,
            chan_3,
            chan_4,
            dir_seg_models=dir_seg_models,
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
            self.ds_train, self.ds_valid, self.ds_extra_valid, classes=1
        )
        self.y.get_segmentation(curve, min_w, max_w, param_curve)

    def prepare_output_inversion(self, N_hours_prec: int = 1):
        """Prepare output object for inversion."""
        self.y = Output_train(
            self.ds_train, self.ds_valid, self.ds_extra_valid, classes=1
        )
        self.y.get_inversion(N_hours_prec=N_hours_prec)


@dataclass
class Input_eval:
    """Prepare and store train and valid inputs."""

    ds: xr.Dataset
    chan_0: str
    chan_1: str = "None"
    chan_2: str = "None"
    chan_3: str = "None"
    chan_4: str = "None"
    dir_seg_models: str = "None"
    noise_level: float = 0.7

    def __post_init__(self):

        self.list_chans = [
            self.chan_0,
            self.chan_1,
            self.chan_2,
            self.chan_3,
            self.chan_4,
        ]

        filler = Input_filler(
            self.dir_seg_models,
            self.noise_level,
        )

        self.eval = filler.fill_data(
            self.ds,
            self.list_chans,
        )
        self.fields_input_shape = list(self.eval.shape[1:])


@dataclass
class Output_eval:
    """Prepare and store train and valid outputs."""

    ds_eval: xr.Dataset
    classes: int

    def get_segmentation(self, curve, min_w, max_w, param_curve):
        """Get segmentation train and valid."""
        self.eval = get_weighted_plume(self.ds_eval, curve, min_w, max_w, param_curve)
        print("data.y.train.shape", self.eval.shape)

    def get_inversion(self, N_hours_prec):
        """Get inversion train and valid."""
        self.eval = get_emiss(self.ds_eval, N_hours_prec)


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
        dir_seg_models: str = "/cerea_raid/users/dumontj/dev/coco2/dl/res/models",
    ):
        """Prepare input object."""
        self.x = Input_eval(
            self.ds,
            chan_0,
            chan_1,
            chan_2,
            chan_3,
            chan_4,
            dir_seg_models=dir_seg_models,
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
        self.y.get_inversion(N_hours_prec=N_hours_prec)
