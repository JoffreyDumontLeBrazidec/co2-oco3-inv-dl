# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
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


def get_scaler(path_w: str) -> preprocessing.StandardScaler:
    """Get scaler corresponding to a training dataset."""
    scaler = joblib.load(os.path.join(path_w, "scaler.save"))
    return scaler


def extract_valid_dataset(ds: xr.Dataset, ds_inds: dict) -> xr.Dataset:
    """Extract valid dataset from ds"""
    ds_valid = ds.sel(idx_img=ds_inds["valid"])
    return ds_valid


class Eval_Shuffler:
    """Shuffler creator train, validation sets for model evaluation."""

    def __init__(
        self,
        path_train: str,
        path_valid: str,
        train_ratio: float,
        tv_split: str = "regular",
        ds_inds: dict = dict(),
    ):

        if path_train == path_valid:
            N_data = xr.open_dataset(path_train).N_img
            if ds_inds == dict():
                if tv_split == "random":
                    self.make_random_split_indices(train_ratio, N_data)
                elif tv_split == "regular":
                    self.make_regular_split_indices(train_ratio, N_data)
            else:
                self.ds_inds = ds_inds
        else:
            N_img_train = xr.open_dataset(path_train).N_img
            N_img_valid = xr.open_dataset(path_valid).N_img
            self.ds_inds = {
                "train": list(np.arange(0, N_img_train, 1)),
                "valid": list(np.arange(N_img_train, N_img_train + N_img_valid, 1)),
            }

    def make_random_split_indices(self, train_ratio: float, N_data: int):
        """Make random list of train and validation indices."""
        shuffle_indices = np.random.permutation(N_data)
        N_train = int(np.floor(train_ratio * N_data))
        self.ds_inds = {
            "train": list(shuffle_indices[0:N_train]),
            "valid": list(shuffle_indices[N_train:]),
        }

    def make_regular_split_indices(self, train_ratio: float, N_data: int):
        """Make regular list of train and validation indices."""
        duration_valid_cycle = 2 * 24
        duration_cycle = duration_valid_cycle / (1 - train_ratio)
        burn_in = 24
        beg = np.random.choice(np.arange(burn_in, duration_cycle))
        N_cycles = math.ceil(N_data / duration_cycle)
        valid_cycle_begs = np.linspace(beg, N_data - duration_cycle, N_cycles).astype(
            int
        )
        inds_valid = np.concatenate(
            [np.arange(i, i + duration_valid_cycle) for i in valid_cycle_begs]
        )
        inds_train = np.delete(np.arange(N_data), inds_valid)
        self.ds_inds = {
            "train": list(inds_train),
            "valid": list(inds_valid),
        }

    def train_valid_split(self, data):
        """Create train, validation sets from data."""
        data_train = data[self.ds_inds["train"]]
        data_valid = data[self.ds_inds["valid"]]

        return list((data_train, data_valid))


@dataclass
class Xco2:
    """xco2 pure or noisy container."""

    ds: xr.Dataset
    noise_level: float = 0.7

    def get_pure(self) -> np.ndarray:
        """Return pure xco2 field related to ds."""
        return np.expand_dims(self.ds.xco2.values, -1)

    def get_noisy(self) -> np.ndarray:
        """Return noisy xco2 field related to ds."""
        xco2 = np.expand_dims(self.ds.xco2.values, -1)
        noise = self.noise_level * np.random.randn(*xco2.shape)
        xco2 = xco2 + noise
        return xco2


@dataclass
class No2:
    """nco2 pure or noisy container."""

    ds: xr.Dataset
    noise_level: str = "field"

    def get_pure(self) -> np.ndarray:
        """Return pure nco2 field related to ds."""
        return np.expand_dims(self.ds.nco2.values, -1)

    def get_noisy(self) -> np.ndarray:
        """Return noisy nco2 field related to ds."""
        no2 = np.expand_dims(self.ds.no2.values, -1)
        no2_noisy = no2 + np.random.randn(*no2.shape) * no2
        return no2_noisy


@dataclass
class Segmentations_pred:
    """To return segmentation predictions"""

    dir_seg_models: str
    name_seg_model: str
    scaler: preprocessing.StandardScaler
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
        xco2_cont = Xco2(self.ds, self.noise_level)
        xco2_noisy = xco2_cont.get_noisy()
        if self.model.layers[0].input_shape[0][-1] == 1:
            inputs = xco2_noisy
        elif self.model.layers[0].input_shape[0][-1] == 2:
            no2_cont = No2(self.ds)
            no2_noisy = no2_cont.get_noisy()
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
class Input_train:
    """Prepare and store train and valid inputs."""

    ds: xr.Dataset
    eval_shuffler: Eval_Shuffler
    channel_0: str
    channel_1: str = "None"
    channel_2: str = "None"
    dir_seg_models: str = "None"
    noise_level: float = 0.7

    def __post_init__(self):

        data = self.fill_data()

        self.eval_split(data, self.eval_shuffler)

        self.get_scaler()

        self.standardise()

    def fill_data(self) -> np.ndarray:
        """Fill input data according to channel_0,1,2"""
        data = self.fill_channel(self.channel_0)
        for channel in [x for x in [self.channel_1, self.channel_2] if x != "None"]:
            data = np.concatenate((data, self.fill_channel(channel)), axis=-1)
        self.fields_input_shape = list(data.shape[1:])
        return data

    def fill_channel(self, channel) -> np.ndarray:
        """Return array depending on channel type specified."""
        if channel == "xco2":
            xco2_cont = Xco2(self.ds, self.noise_level)
            data_channel = xco2_cont.get_noisy()
        elif channel == "no2":
            no2_cont = No2(self.ds)
            data_channel = no2_cont.get_noisy()
        elif channel.startswith("seg"):
            seg_cont = Segmentations_pred(
                self.dir_seg_models, channel, scaler=self.scaler, ds=self.ds
            )
            data_channel = seg_cont.get_seg_predictions()
        else:
            sys.exit()

        return data_channel

    def eval_split(self, data, eval_shuffler):
        """Split data in train and valid with eval_shuffler"""
        [self.train, self.valid] = eval_shuffler.train_valid_split(data)

    def get_scaler(self):
        """Create scaler if self.scaler==None."""
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(self.train.reshape(-1, self.train.shape[-1]))

    def standardise(self):
        """Standardise data according to f_train or given scaler."""
        self.train = np.array(
            self.scaler.transform(self.train.reshape(-1, self.train.shape[-1]))
        ).reshape(self.train.shape)

        self.valid = np.array(
            self.scaler.transform(self.valid.reshape(-1, self.valid.shape[-1]))
        ).reshape(self.valid.shape)

        print("data.x.train.shape", self.train.shape)


@dataclass
class Output_train:
    """Prepare and store train and valid outputs."""

    def get_plume(self, ds, eval_shuffler):
        """Get train, valid plume."""
        plume = np.array(ds.plume.values, dtype=float)
        [
            self.plume_train,
            self.plume_valid,
        ] = eval_shuffler.train_valid_split(plume)

    def get_segmentation(
        self,
        ds: xr.Dataset,
        eval_shuffler: Eval_Shuffler,
        curve: str,
        min_w: float,
        max_w: float,
        param_curve: float,
    ):
        """Get modified plume matrices label output."""
        self.classes = 1
        y_data = calculate_weighted_plume(
            np.array(ds.plume.values, dtype=float), min_w, max_w, curve, param_curve
        )
        self.get_eval_labels(y_data, eval_shuffler)

    def get_inversion(
        self, ds: xr.Dataset, eval_shuffler: Eval_Shuffler, N_hours_prec: int
    ):
        """Get emissions vector label output."""
        self.classes = N_hours_prec
        emiss = np.array(ds.emiss.values, dtype=float)
        emiss = emiss[:, : self.classes]
        self.get_eval_labels(emiss, eval_shuffler)

    def get_eval_labels(self, data: np.ndarray, eval_shuffler: Eval_Shuffler):
        """Get train, valid or test label data."""
        [
            self.train,
            self.valid,
        ] = eval_shuffler.train_valid_split(data)
        print("data.y.train.shape", self.train.shape)


@dataclass
class Data_train:
    """Object for containing Input and Output data and all other informations."""

    eval_shuffler: Eval_Shuffler
    path_train_nc: str
    path_valid_nc: str

    def __post_init__(self):
        if self.path_train_nc == self.path_valid_nc:
            self.ds = xr.open_dataset(self.path_train_nc)
        else:
            self.ds = xr.concat(
                (
                    xr.open_dataset(self.path_train_nc),
                    xr.open_dataset(self.path_valid_nc),
                ),
                dim="img",
            )

    def prepare_input(
        self,
        channel_0: str,
        channel_1: str = "None",
        channel_2: str = "None",
        dir_seg_models: str = "None",
    ):
        """Prepare input object."""
        self.x = Input_train(
            self.ds,
            self.eval_shuffler,
            channel_0,
            channel_1,
            channel_2,
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
        self.y = Output_train()
        self.y.get_segmentation(
            self.ds, self.eval_shuffler, curve, min_w, max_w, param_curve
        )

    def prepare_output_inversion(self, N_hours_prec: int = 1):
        """Prepare output object for inversion."""
        self.y = Output_train()
        self.y.get_inversion(self.ds, self.eval_shuffler, N_hours_prec)


@dataclass
class Input_eval:
    """Prepare and store train and valid inputs."""

    ds: xr.Dataset
    scaler: preprocessing.StandardScaler
    channel_0: str
    channel_1: str = "None"
    channel_2: str = "None"
    dir_seg_models: str = "None"
    noise_level: float = 0.7

    def __post_init__(self):

        self.eval = self.fill_data()
        self.standardise()

    def fill_data(self) -> np.ndarray:
        """Fill input data according to channel_0,1,2"""
        data = self.fill_channel(self.channel_0)
        for channel in [x for x in [self.channel_1, self.channel_2] if x != "None"]:
            data = np.concatenate((data, self.fill_channel(channel)), axis=-1)

        self.fields_input_shape = list(data.shape[1:])
        return data

    def fill_channel(self, channel: str) -> np.ndarray:
        """Return array depending on channel type specified."""
        if channel == "xco2":
            xco2_cont = Xco2(self.ds, self.noise_level)
            data_channel = xco2_cont.get_noisy()
        elif channel == "no2":
            no2_cont = No2(self.ds)
            data_channel = no2_cont.get_noisy()
        elif channel.startswith("seg"):
            seg_cont = Segmentations_pred(
                self.dir_seg_models, channel, scaler=self.scaler, ds=self.ds
            )
            data_channel = seg_cont.get_seg_predictions()
        else:
            sys.exit()

        return data_channel

    def standardise(self) -> None:
        """Standardise data according to f_train or given scaler."""
        self.eval = np.array(
            self.scaler.transform(self.eval.reshape(-1, self.eval.shape[-1]))
        ).reshape(self.eval.shape)

        print("data.x.eval.shape", self.eval.shape)


@dataclass
class Output_eval:
    """Prepare and store valid or test outputs."""

    def get_plume(self, ds: xr.Dataset):
        """Get plume."""
        plume = np.array(ds.plume.values, dtype=float)
        self.plume_eval = plume

    def get_segmentation(
        self, ds: xr.Dataset, curve: str, min_w: float, max_w: float, param_curve: float
    ):
        """Get modified plume matrices label output."""
        y_data = calculate_weighted_plume(
            np.array(ds.plume.values, dtype=float), min_w, max_w, curve, param_curve
        )
        self.eval = y_data

    def get_inversion(self, ds: xr.Dataset, N_hours_prec: int):
        """Get emissions vector label output."""
        self.classes = N_hours_prec
        emiss = np.array(ds.emiss.values, dtype=float)
        emiss = emiss[:, : self.classes]
        self.eval = emiss


@dataclass
class Data_eval:

    path_eval_nc: str
    ds_inds: dict = field(default_factory=lambda: dict())
    region_extrapol: bool = False

    def __post_init__(self):
        self.ds = xr.open_dataset(self.path_eval_nc)
        if ("valid" in self.path_eval_nc) & (self.region_extrapol == False):
            self.ds = extract_valid_dataset(self.ds, self.ds_inds)

    def prepare_input(
        self,
        scaler: preprocessing.StandardScaler,
        channel_0: str,
        channel_1: str = "None",
        channel_2: str = "None",
        dir_seg_models: str = "None",
    ):
        """Prepare input object."""
        self.x = Input_eval(
            self.ds,
            scaler,
            channel_0,
            channel_1,
            channel_2,
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
        self.y = Output_eval()
        self.y.get_segmentation(self.ds, curve, min_w, max_w, param_curve)

    def prepare_output_inversion(self, N_hours_prec: int = 1):
        """Prepare output object for inversion."""
        self.y = Output_eval()
        self.y.get_inversion(self.ds, N_hours_prec)
