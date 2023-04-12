#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import numpy as np
import xarray as xr
import pandas as pd
import os
import sys
from sklearn import preprocessing
from collections import namedtuple
from dataclasses import dataclass

from treeconfigparser import TreeConfigParser


class Eval_Shuffler:
    """Shuffle for model evaluation."""

    def __init__(
        self, train_ratio: float, valid_ratio: float, N_data: int, shuffle_indices=None
    ):

        if shuffle_indices == None:
            shuffle_indices = np.random.permutation(N_data)
        else:
            shuffle_indices = shuffle_indices

        N_train = int(np.floor(train_ratio * N_data))
        N_valid = int(np.floor(valid_ratio * N_data))

        self.ds_inds = {
            "train": list(shuffle_indices[0:N_train]),
            "valid": list(shuffle_indices[N_train : N_train + N_valid]),
            "test": list(shuffle_indices[N_train + N_valid :]),
        }

    def train_valid_test_split(self, data):
        """Create train, validation, test sets from data."""

        data_train = data[self.ds_inds["train"]]
        data_valid = data[self.ds_inds["valid"]]
        data_test = data[self.ds_inds["test"]]

        return list((data_train, data_valid, data_test))


class Input:
    """Object for storing train, valid, and test inputs."""
    def __init__(
        self,
        ds: xr.Dataset,
        eval_shuffler: Eval_Shuffler,
        config: TreeConfigParser,
        scaler,
    ):

        xco2 = np.expand_dims(ds.xco2.values, -1)
        xco2 = self.add_noise(
            xco2,
            config.get_bool("data.input.xco2.noise.bool"),
            config.get_float("data.input.xco2.noise.level"),
        )
        
        [f_train, f_valid, f_test] = eval_shuffler.train_valid_test_split(xco2)
        [f_train, f_valid, f_test] = self.standardise(f_train, f_valid, f_test, scaler)

        self.train = [f_train]
        self.valid = [f_valid]
        self.test = [f_test]
     
        # f_data = self.add_supp_fields_input(f_data)

        if config.get("data.output.label.choice") == "regression":
            self.superpose_field_in_three_channels()
            self.fields_input_shape = [ds["y"].shape[0], ds["x"].shape[0], 3]
        else:
            self.fields_input_shape = [ds["y"].shape[0], ds["x"].shape[0], 1]

    def standardise(self, f_train, f_valid, f_test, scaler):
        """Standardise data according to f_train or given scaler."""
        if scaler == None:
            scaler = preprocessing.StandardScaler()
            scaler.fit(f_train.reshape(-1, f_train.shape[-1]))

        f_train = scaler.transform(f_train.reshape(-1, f_train.shape[-1])).reshape(
            f_train.shape
        )
        f_valid = scaler.transform(f_valid.reshape(-1, f_valid.shape[-1])).reshape(
            f_valid.shape
        )
        f_test = scaler.transform(f_test.reshape(-1, f_test.shape[-1])).reshape(
            f_test.shape
        )        
        return [f_train, f_valid, f_test]

    def add_noise(self, xco2, noise_bin, noise_level):
        """Add 1ppm var noise to xco2 field."""
        if noise_bin:
            noise = noise_level * np.random.randn(*xco2.shape).astype(xco2.dtype)
            xco2 = xco2 + noise
        return xco2

    def superpose_field_in_three_channels(self):
        """Used for CNN which take 3D images as inputs."""
        if self.train[0].shape[-1] == 1:
            self.train[0] = np.tile(self.train[0], (1, 1, 1, 3))
            self.valid[0] = np.tile(self.valid[0], (1, 1, 1, 3))
            self.test[0] = np.tile(self.test[0], (1, 1, 1, 3))
        else:
            print(
                "Shape field is:",
                self.train[0].shape[-1],
                "and should be 1 to be reshaped to 3",
            )
            sys.exit()


@dataclass
class Output:
    """Object for storing train, valid, and test outputs."""

    labelling: str

    def get_trace(self, ds, eval_shuffler):
        """Get train, valid, test trace."""
        trace = np.array(ds.trace.values, dtype=np.float32)
        [
            self.trace_train,
            self.trace_valid,
            self.trace_test,
        ] = eval_shuffler.train_valid_test_split(trace)

    def get_presence(self, ds, eval_shuffler, config):
        """Get presence vector label output."""
        self.classes = 1
        y_data = np.array(ds.ppresence.values, dtype=np.float32)
        self.get_train_valid_test(y_data, eval_shuffler)

    def get_segmentation_with_bin(self, ds, eval_shuffler, config):
        """Get trace binary matrices label output."""
        self.classes = 1
        y_data = np.array(ds.pixels_plume.values, dtype=np.float32)
        y_data = np.expand_dims(y_data, axis=-1).shape
        self.get_train_valid_test(y_data, eval_shuffler)

    def get_segmentation_with_trace(self, ds, eval_shuffler, config):
        """Get modified trace matrices label output."""
        self.classes = 1

        trace = np.array(ds.trace.values, dtype=np.float32)
        min_w = config.get_float("data.output.label.weight.min")
        max_w = config.get_float("data.output.label.weight.max")
        threshold_min = ds.attrs["thresh_val"]
        N_data = ds.N_img

        y_min = np.repeat([threshold_min], N_data).reshape(N_data, 1, 1)
        y_max = np.quantile(trace, q=0.99, axis=(1, 2)).reshape(N_data, 1, 1)
        weight_min = np.repeat([min_w], N_data).reshape(N_data, 1, 1)
        weight_max = np.repeat([max_w], N_data).reshape(N_data, 1, 1)
        pente = (weight_max - weight_min) / (y_max - y_min)
        b = weight_min - pente * y_min

        y_data = pente * trace + b * np.where(trace > 0, 1, 0)
        y_data = np.where(y_data < max_w, y_data, max_w)

        y_data = np.expand_dims(y_data, axis=-1).shape
        self.get_train_valid_test(y_data, eval_shuffler)

    def get_pixel_wise_regression(self, ds, eval_shuffler, config):
        """Get trace matrices label output."""
        self.classes = 1
        trace = np.array(ds.trace.values, dtype=np.float32)
        self.get_train_valid_test(trace, eval_shuffler)

    def get_regression(self, ds, eval_shuffler, config):
        """Get emissions vector label output."""
        self.classes = config.get_int("data.output.label.N_hours_prec")
        emiss = np.array(ds.emiss.values, dtype=np.float32)
        emiss = emiss[:, : self.classes]
        self.get_train_valid_test(emiss, eval_shuffler, config)
        
    def get_train_valid_test(self, data, eval_shuffler, config):
        """Get train, valid, test according to data."""
        [
            self.train,
            self.valid,
            self.test,
        ] = eval_shuffler.train_valid_test_split(data)
        
    def get_label(self, ds, eval_shuffler, config):
        """Get label with method according to labelling."""
        method = getattr(self, "get_" + self.labelling)
        args = [ds, eval_shuffler, config]
        method(*args)


@dataclass
class Data:
    """Object for containing Input and Output data and all other informations."""
    config: TreeConfigParser()
    shuffle_indices: np.ndarray = None

    def __post_init__(self):
        self.name_dataset = self.config.get("data.directory.name")
        self.dir_dataset = os.path.join(
            self.config.get("data.directory.main"), self.name_dataset
        )

        ds = xr.open_dataset(os.path.join(self.dir_dataset, "dataset.nc"))
        self.N_data = ds.N_img
        self.eval_shuffler = Eval_Shuffler(
            self.config.get_float("data.training_ratio"),
            self.config.get_float("data.validation_ratio"),
            self.N_data,
            self.shuffle_indices,
        )

    def prepare_input(self, scaler=None):
        """Prepare input object."""
        ds = xr.open_dataset(os.path.join(self.dir_dataset, "dataset.nc"))
        self.x = Input(ds, self.eval_shuffler, self.config, scaler)

        print("data.inp.train.shape[0]", self.x.train[0].shape)

    def prepare_output(self):
        """Prepare output object."""
        ds = xr.open_dataset(os.path.join(self.dir_dataset, "dataset.nc"))
        self.y = Output(self.config.get("data.output.label.choice"))
        self.y.get_label(ds, self.eval_shuffler, self.config)
        print("data.out.train.shape", self.y.train.shape)

    """
    # ------------------------------------------------------------------------
    # add_supp_fields_input
    def add_supp_fields_input(self, f_data):
        
        if self.config.get("data.input.dynamics.format") == "field":
            self.N_FD = self.config.get_int("data.input.dynamics.fields.number.current")
        else:
            self.N_FD = 0
        if self.config.get("data.input.winds.format") == "field":
            self.N_FW = self.config.get_int("data.input.winds.fields.number.current")
        else:
            self.N_FW = 0
            
        self.N_input_channels = 1 + self.N_FD + self.N_FW
        self.fields_input_shape = (self.Ny, self.Nx, self.N_input_channels)
        expanded_f_data = np.empty(
            (self.N_data, self.N_input_channels, self.Ny, self.Nx), dtype=np.float32
        )
        expanded_f_data[:, 0:1, :, :] = f_data
        
        if self.N_FD > 0:
            expanded_f_data[:, 1 : 1 + self.N_FD, :, :] = self.ds.dynamics.sel(
                dyn_t=slice(0, self.N_FD)
            ).values

        if self.N_FW > 0:
            expanded_f_data[
                :, 1 + self.N_FD : 1 + self.N_FD + self.N_FW, :, :
            ] = self.ds.winds.sel(wind_t=slice(0, self.N_FW)).values
            
        return expanded_f_data

    # ------------------------------------------------------------------------
    # prepare_vector_input
    def prepare_vector_input(self):

        # input vectors
        ## time
        self.include_time = self.config.get_bool("data.input.time")
        if self.include_time:
            self.size_time = 24
        else:
            self.size_time = 0

        ## scale
        self.include_scale = self.config.get_bool("data.input.scale")
        if self.include_scale:
            self.size_scale = 4
        else:
            self.size_scale = 0

        ## winds
        if self.config.get("data.input.winds.format") == "summary":
            self.size_SW = self.config.get_int("data.input.winds.summary.size")
        else:
            self.size_SW = 0

        self.size_input_vector = self.size_time + self.size_SW + self.size_scale
        self.vector_input_shape = (self.size_input_vector, 1)

        if self.size_input_vector > 0:

            v_data = np.empty((self.N_data, self.size_input_vector), dtype=np.float32)
            fill_pos_start = 0
            fill_pos_end = 0

            # Time
            if self.include_time:
                hours = self.ds.v_oneHotHour.values
                fill_pos_start = fill_pos_end
                fill_pos_end = fill_pos_start + self.size_time
                v_data[:, fill_pos_start:fill_pos_end] = hours

            # Winds
            if self.winds == "summary":
                v_winds = self.ds.v_winds.values
                fill_pos_start = fill_pos_end
                fill_pos_end = fill_pos_start + self.size_SW
                v_data[:, fill_pos_start:fill_pos_end] = v_winds

            # Scale
            if self.include_scale:
                scales = self.ds.shape_cropping.values
                fill_pos_start = fill_pos_end
                fill_pos_end = fill_pos_start + self.size_scale
                v_data[:, fill_pos_start:fill_pos_end] = scales

            [v_train, v_valid, v_test] = self.eval_shuffler.train_valid_test_split(
                v_data
            )

            ## Norm
            if self.input_norm == "standardisation":
                scaler = preprocessing.StandardScaler()
                v_train = scaler.fit_transform(v_train)
                v_valid = scaler.transform(v_valid)
                v_test = scaler.transform(v_test)
            elif self.input_norm == "none":
                pass
            else:
                print("Normalisation method poorly defined")
                sys.exit()

            ## append in final input (x)
            self.x_train.append(v_train)
            self.x_valid.append(v_valid)
            self.x_test.append(v_test)
    """


# __________________________________________________________
