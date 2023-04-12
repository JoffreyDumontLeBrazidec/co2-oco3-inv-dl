# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import pickle
import shutil
import sys

import joblib
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing


class Saver:
    """Saver of all results relevant to CNN model training experience."""

    def __init__(self, cfg: DictConfig):
        """Prepare directory to store results of the experiments."""
        if cfg.sweep == "false":
            self.dir_save_model = os.path.join(cfg.dir_res, cfg.exp_name)
        elif cfg.sweep == "true":
            self.dir_save_model = os.path.join(cfg.dir_res, cfg.fifou)
        else:
            sys.exit()
        self.dir_save_model = cfg.save
        # if os.path.exists(self.dir_save_model):
        # shutil.rmtree(self.dir_save_model)
        # os.makedirs(self.dir_save_model)
        self.save_cfg(cfg)
        print("2", os.getcwd())

    def save_cfg(self, cfg: DictConfig):
        """Save config file."""
        path_yaml = os.path.join(self.dir_save_model, "config.yaml")
        OmegaConf.save(config=cfg, f=path_yaml, resolve=True)

    def save_model_and_weights(self, model: tf.keras.Model):
        """Save model and weights using keras built_in functions."""
        print("1", os.getcwd())
        model.save(os.path.join(self.dir_save_model, "weights_model.h5"))

    def save_data_shuffle_indices(self, ds_indices: dict):
        """Save shuffle indices."""
        with open(os.path.join(self.dir_save_model, "tv_inds.pkl"), "wb") as f:
            pickle.dump(ds_indices, f)

    def save_input_scaler(self, scaler: preprocessing.StandardScaler):
        """Save input data sklearn scaler."""
        joblib.dump(scaler, os.path.join(self.dir_save_model, "scaler.save"))
