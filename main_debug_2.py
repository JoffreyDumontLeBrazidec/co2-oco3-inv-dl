# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import sys

import hydra
import tensorflow as tf
from omegaconf import DictConfig, OmegaConf

from model_training import Model_training_manager
from saver import Saver


@hydra.main(config_path="cfg", config_name="config")
def main_train(cfg: DictConfig):
    # orig_cwd = hydra.utils.get_original_cwd()
    sav = Saver(cfg)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(16, input_dim=20))
    sav.save_model_and_weights(model)


if __name__ == "__main__":
    main_train()
