# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from sklearn import preprocessing
from collections import namedtuple
from dataclasses import dataclass
import math
import joblib

from treeconfigparser import TreeConfigParser
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl")
from include.loss import pixel_weighted_cross_entropy, calculate_weighted_plume

def get_scaler(path_w: str):
    """Get scaler corresponding to a training dataset."""
    scaler = joblib.load(os.path.join(path_w, "scaler.save"))
    return scaler


class Eval_Shuffler:
    """Shuffler creator train, validation sets for model evaluation."""

    def __init__(
        self, path_train, path_valid, train_ratio: float, N_data: int, ds_inds=None, tv_split: str = "regular"
    ):
        if path_train == path_valid:
            if ds_inds is None:
                if tv_split == "random":
                    self.make_random_split_indices(train_ratio, N_data)
                elif tv_split == "regular":
                    self.make_regular_split_indices(train_ratio, N_data)
            else:
                self.ds_inds = ds_inds
        else:
            N_img_train = xr.open_dataset(path_train).N_img
            N_img_valid = xr.open_dataset(path_valid).N_img
            ds_inds = {
            "train": list(np.arange(0, N_img_train, 1)),
            "valid": list(np.arange(N_img_train, N_img_train+N_img_valid, 1)),
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


class Input:
    """Prepare and store tvt inputs."""

    def __init__(
        self,
        ds: xr.Dataset,
        eval_shuffler: Eval_Shuffler,
        config: TreeConfigParser,
        scaler,
        mode: str,
        dir_seg_models: str = None,
        supp_inputs: list = [],
    ):

        data = self.fill_channel_0(
            ds, config.get_float("data.input.xco2.noise.level"), mode
        )
        data = self.fill_channels_12(
            data,
            mode,
            supp_inputs,
            dir_seg_models,
            ds,
        )
        self.fields_input_shape = data.shape[1:]

        self.eval_split(data, eval_shuffler, mode)

        self.get_scaler(scaler, mode)

        self.standardise(mode)

    def fill_channel_0(self, ds: xr.Dataset, noise_level: np.float32, mode: str):
        """Add Xppm var noise to xco2 field."""
        if mode == "train":
            xco2 = np.expand_dims(ds.xco2.values, -1)
            noise = noise_level * np.random.randn(*xco2.shape).astype(xco2.dtype)
            xco2 = xco2 + noise
        elif mode == "test":
            xco2 = np.expand_dims(ds.xco2_noisy.values, -1)
        return xco2

    def fill_channels_12(
        self,
        data: np.ndarray,
        mode: str,
        supp_inputs: list,
        dir_seg_models: str,
        ds: xr.Dataset,
    ):
        """Fill channels 1 and 2 of input field data."""

        seg_fields_inds = [
            supp_inputs.index(inp) for inp in supp_inputs if inp.startswith("seg")
        ]
        for seg_ind in seg_fields_inds:

            model = keras.models.load_model(
                os.path.join(dir_seg_models, f"{supp_inputs[seg_ind]}", "weights_cp_best.h5"),
                compile=False,
            )
            model.compile("adam", loss=pixel_weighted_cross_entropy)
            scaler = get_scaler(os.path.join(dir_seg_models, supp_inputs[seg_ind]))
            x_seg = get_seg_model_pred_on_ds(model, scaler, ds)
            data = np.concatenate((data, x_seg), axis=-1)
 
        if "NO2" in supp_inputs:
            no2 = np.expand_dims(ds.no2.values, -1)
            no2_noisy = no2 + np.random.randn(*no2.shape) * no2
            data = np.concatenate((data, no2_noisy), axis=-1)
            
        if data.shape[-1] == 2:
            data = np.concatenate((data, data[:,:,:,0:1]), axis=-1)

        return data

    def eval_split(self, data, eval_shuffler, mode):
        """Split data in train and valid with eval_shuffler if mode=train"""
        if mode == "train":
            [self.train, self.valid] = eval_shuffler.train_valid_split(data)
        elif mode == "test":
            self.test = data

    def get_scaler(self, scaler, mode):
        """Create scaler if self.scaler==None."""
        if scaler == None:
            if mode == "train":
                self.scaler = preprocessing.StandardScaler()
                self.scaler.fit(self.train.reshape(-1, self.train.shape[-1]))
            elif mode == "test":
                print("Test mode: scaler must be given")
                sys.exit()
        else:
            self.scaler = scaler

    def standardise(self, mode):
        """Standardise data according to f_train or given scaler."""
        if mode == "train":
            self.train = self.scaler.transform(
                self.train.reshape(-1, self.train.shape[-1])
            ).reshape(self.train.shape)
            self.valid = self.scaler.transform(
                self.valid.reshape(-1, self.valid.shape[-1])
            ).reshape(self.valid.shape)
            
            print("data.x.train.shape", self.train.shape)

        elif mode == "test":
            self.test = self.scaler.transform(
                self.test.reshape(-1, self.test.shape[-1])
            ).reshape(self.test.shape)
            print("data.x.test.shape", self.test.shape)

class Input_train:
    """Prepare and store train and valid inputs."""

    def __init__(
        self,
        channel_0,
        channel_1: str = None,
        channel_2: str = None,
        ds: xr.Dataset,
        eval_shuffler: Eval_Shuffler,
        scaler = None,
        dir_seg_models: str = None,
    ):

        data = self.fill_data()
        
        self.fields_input_shape = data.shape[1:]

        self.eval_split(data, eval_shuffler)

        self.get_scaler(scaler)

        self.standardise()

    def fill_data(self, ...):
        
        data = self.fill_channel(self.channel_0)
        for channel in [x for x in [self.channel_1, self.channel_2] if x]:
            data = np.concatenate((data, self.fill_channel(channel)), axis=-1)
                                
    def fill_channel(self, channel):
        """Return array depending on channel type specified."""
        if channel == "xco2":
            xco2_cont = Xco2(self.ds, self.noise_level)
            data_channel =  Xco2.get_noisy()
        elif channel == "no2":
            no2_cont = No2(self.ds)
            data_channel = No2.get_noisy()
        elif channel.startswith("seg"):
            seg_cont = Segmentations_pred(self.dir_seg_models, channel, scaler = self.scaler, ds = self.ds)
            data_channel = seg_cont.get_seg_predictions()
                        
    def eval_split(self, data, eval_shuffler, mode):
        """Split data in train and valid with eval_shuffler if mode=train"""
        [self.train, self.valid] = eval_shuffler.train_valid_split(data)


    def get_scaler(self, scaler, mode):
        """Create scaler if self.scaler==None."""
        if scaler == None:
            self.scaler = preprocessing.StandardScaler()
            self.scaler.fit(self.train.reshape(-1, self.train.shape[-1]))
        else:
            self.scaler = scaler

    def standardise(self, mode):
        """Standardise data according to f_train or given scaler."""
        self.train = self.scaler.transform(
            self.train.reshape(-1, self.train.shape[-1])
        ).reshape(self.train.shape)
        self.valid = self.scaler.transform(
            self.valid.reshape(-1, self.valid.shape[-1])
        ).reshape(self.valid.shape)

        print("data.x.train.shape", self.train.shape)

@dataclass
class Output:
    """Prepare and store tvt outputs."""

    labelling: str
    mode: str

    def get_plume(self, ds, eval_shuffler):
        """Get train, valid plume."""
        plume = np.array(ds.plume.values, dtype=np.float32)
        if self.mode == "train":
            [
                self.plume_train,
                self.plume_valid,
            ] = eval_shuffler.train_valid_split(plume)
        elif self.mode == "test":
            self.plume_test = plume

    def get_presence(self, ds, eval_shuffler, config):
        """Get presence vector label output."""
        self.classes = 1
        y_data = np.array(ds.ppresence.values, dtype=np.float32)
        self.get_eval_labels(y_data, eval_shuffler)

    def get_segmentation(self, ds, eval_shuffler, config):
        """Get modified plume matrices label output."""
        self.classes = 1
        plume = np.array(ds.plume.values, dtype=np.float32)
        min_w = config.get_float("data.output.label.weight.min")
        max_w = config.get_float("data.output.label.weight.max")
        weighting_curve = config.get("data.output.label.weight.curve")
        weighting_param = config.get_float("data.output.label.weight.param")

        y_data = calculate_weighted_plume(
            plume, min_w, max_w, weighting_curve, weighting_param
        )

        self.get_eval_labels(y_data, eval_shuffler)

    def get_segmentation_weighted(self, ds, eval_shuffler, config):
        """Get modified plume matrices label output."""
        self.classes = 1
        plume = np.array(ds.plume.values, dtype=np.float32)
        min_w = config.get_float("data.output.label.weight.min")
        max_w = config.get_float("data.output.label.weight.max")
        threshold_min = 0.05
        N_data = ds.N_img

        y_min = np.repeat([threshold_min], N_data).reshape(N_data, 1, 1)
        y_max = np.quantile(plume, q=0.99, axis=(1, 2)).reshape(N_data, 1, 1)
        weight_min = np.repeat([min_w], N_data).reshape(N_data, 1, 1)
        weight_max = np.repeat([max_w], N_data).reshape(N_data, 1, 1)
        pente = (weight_max - weight_min) / (y_max - y_min)
        b = weight_min - pente * y_min

        y_data = pente * plume + b * np.where(plume > 0, 1, 0)
        y_data = np.where(y_data < max_w, y_data, max_w)

        y_data = np.expand_dims(y_data, axis=-1)
        self.get_eval_labels(y_data, eval_shuffler)

    def get_pixel_wise_regression(self, ds, eval_shuffler, config):
        """Get plume matrices label output."""
        self.classes = 1
        plume = np.array(ds.plume.values, dtype=np.float32)
        self.get_eval_labels(plume, eval_shuffler)

    def get_inversion(self, ds, eval_shuffler, config):
        """Get emissions vector label output."""
        self.classes = config.get_int("data.output.label.N_hours_prec")
        emiss = np.array(ds.emiss.values, dtype=np.float32)
        emiss = emiss[:, : self.classes]
        self.get_eval_labels(emiss, eval_shuffler)

    def get_eval_labels(self, data, eval_shuffler):
        """Get train, valid or test label data."""
        if self.mode == "train":
            [
                self.train,
                self.valid,
            ] = eval_shuffler.train_valid_split(data)
            print("data.y.train.shape", self.train.shape)
        elif self.mode == "test":
            self.test = data
            print("data.y.test.shape", self.test.shape)

    def get_label(self, ds, eval_shuffler, config):
        """Get label with method according to labelling."""
        method = getattr(self, "get_" + self.labelling)
        args = [ds, eval_shuffler, config]
        method(*args)


@dataclass
class Data:
    """Object for containing Input and Output data and all other informations."""

    cfg: DictConfig
    ds_inds: dict = None
    mode: str = "train"

    def __post_init__(self):

        name_dataset = {
            "train": "2d_train_valid_dataset.nc",
            "test": "2d_test_dataset.nc",
        }[self.mode]

        self.path_dataset = os.path.join(
            self.config.get("data.directory.main"),
            self.config.get("data.directory.name"),
            name_dataset,
        )

        self.eval_shuffler = Eval_Shuffler(
            self.config.get_float("data.training_ratio"),
            xr.open_dataset(self.path_dataset).N_img,
            self.ds_inds,
            self.config.get("data.tv_split"),
        )

    def prepare_input(self, scaler=None):
        """Prepare input object."""
        self.x = Input(
            xr.open_dataset(self.path_dataset),
            self.eval_shuffler,
            self.config,
            scaler,
            self.mode,
            self.config.get("dir_seg_models"),
            self.config.get_stringlist("data.input.supps.list"),
        )

    def prepare_output(self):
        """Prepare output object."""
        self.y = Output(self.config.get("data.output.label.choice"), self.mode)
        self.y.get_label(
            xr.open_dataset(self.path_dataset), self.eval_shuffler, self.config
        )

class Data_test

@dataclass
class Data_train:
    """Object for containing Input and Output data and all other informations."""

    path_train_nc: str
    path_valid_nc: str
    train_ratio: np.float32 = 0.85
    split_type: str = "regular"
    ds_inds: dict = None

    def __post_init__(self):
        self.eval_shuffler = Eval_Shuffler(
            self.train_ratio,
            self.path_train_nc,
            self.path_valid_nc,
            self.ds_inds,
            self.split_type,
        )
        
        if self.path_train_nc == self.path_valid_nc:
            self.ds = xr.open_dataset(self.path_train_nc)
        else:
            self.ds = xr.concat(xr.open_dataset(self.path_train_nc),
                                xr.open_dataset(self.path_valid_nc),
                                dim = "img")

    def prepare_input(self, channel_0: str, channel_1: str = None, channel_2: str = None, dir_seg_models: str):
        """Prepare input object."""
        self.x = Input_train(
            channel_0,
            channel_1,
            channel_2,
            self.ds,
            self.eval_shuffler,
            dir_seg_models = dir_seg_models,
        )

    def prepare_output(self):
        """Prepare output object."""
        self.y = Output(self.config.get("data.output.label.choice"), self.mode)
        self.y.get_label(
            xr.open_dataset(self.path_dataset), self.eval_shuffler, self.config
        )

    
@dataclass
class Xco2:
    """xco2 pure or noisy container."""
    ds: xr.Dataset
    noise_level: np.float32 = 0.7
    
    def get_pure():
        """Return pure xco2 field related to ds."""
        return np.expand_dims(ds.xco2.values, -1)

    def get_noisy():
        """Return noisy xco2 field related to ds."""
        xco2 = np.expand_dims(ds.xco2.values, -1)
        noise = noise_level * np.random.randn(*xco2.shape).astype(xco2.dtype)
        xco2 = xco2 + noise
        return xco2
    
@dataclass
class No2:
    """nco2 pure or noisy container."""
    ds: xr.Dataset
    noise_level: str = "field"
    
    def get_pure():
        """Return pure nco2 field related to ds."""
        return np.expand_dims(ds.nco2.values, -1)

    def get_noisy():
        """Return noisy nco2 field related to ds."""
        no2 = np.expand_dims(ds.no2.values, -1)
        no2_noisy = no2 + np.random.randn(*no2.shape) * no2
        return no2_noisy
    
@dataclass
class Segmentations_pred:
    """To return segmentation predictions"""
    dir_seg_models: str
    name_seg_model: str
    model_optimiser: str = "adam"
    model_loss = pixel_weighted_cross_entropy
    scaler: preprocessing.StandardScaler
    ds: xr.Dataset
    noise_level: np.float32 = 0.7
    
    def __post_init__(self):
        self.model = keras.models.load_model(
            os.path.join(self.dir_seg_models, f"{self.name_seg_model}", "weights_cp_best.h5"),
            compile=False,
        )
        self.model.compile(self.model_optimiser, loss=self.model_loss)
        self.scaler = get_scaler(os.path.join(self.dir_seg_models, self.name_seg_model))
    
    def get_seg_predictions():
        """Return segmentation predictions from Model on Dataset."""
        xco2_cont = Xco2(self.ds, self.noise_level)
        xco2_noisy = Xco2.get_noisy()
        if self.model.layers[0].input_shape[0][-1] == 1:
            inputs = xco2
        elif self.model.layers[0].input_shape[0][-1] == 2:
            no2_cont = No2(self.ds)
            no2_noisy = No2.get_noisy()
            inputs = np.concatenate((xco2_noisy, no2_noisy), axis=-1)     
 
        inputs = self.scaler.transform(inputs.reshape(-1, inputs.shape[-1])).reshape(inputs.shape)
        inputs = tf.convert_to_tensor(inputs, np.float32)
        x_seg = tf.convert_to_tensor(self.model.predict(inputs), np.float32)        

    return x_seg
            
class Input_test