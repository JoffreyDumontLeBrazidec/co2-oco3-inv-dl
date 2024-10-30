# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021-2024
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tensorflow as tf
from hydra import compose, initialize
from hydra.utils import call, instantiate
from icecream import ic
from omegaconf import DictConfig, OmegaConf

import wandb

try:
    import models.reg as rm
except ImportError:
    pass

from wandb.keras import WandbMetricsLogger  # type: ignore

import include.callbacks as callbacks
import include.generators as generators
import include.loss as loss
import include.optimisers as optimisers
import models.seg as sm
from Data import Data_train
from include.trainer import Trainer
from models.preprocessing import CloudsLayer
from saver import Saver


class Model_training_manager:
    """Manager for segmentation, inversion with CNN models."""

    def __init__(self, cfg: DictConfig) -> None:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.info("Starting data preparation...")
        self.prepare_data(cfg)
        logging.info("Data preparation completed.")

        logging.info("Starting model building...")
        self.build_model(cfg)
        self.compile_model(cfg)
        logging.info("Model building completed.")

        logging.info("Starting training preparation...")
        self.prepare_training(cfg)
        logging.info("Training preparation completed.")

        logging.info("Initializing the saver...")
        self.saver = Saver()
        logging.info("Saver initialized.")

    def prepare_data(self, cfg: DictConfig) -> None:
        """Prepare Data inputs to the neural network and outputs (=labels, targets)."""

        print(cfg.data.init.path_train_ds)
        self.data = instantiate(cfg.data.init)

        if cfg.model.type in ["inversion", "uq"]:
            self.data.prepare_input(
                cfg.data.input.chan_0,
                cfg.data.input.chan_1,
                cfg.data.input.chan_2,
                cfg.data.input.chan_3,
                cfg.data.input.chan_4,
                cfg.data.input.clouds_threshold,
                cfg.data.input.dir_clouds_array,
                cfg.data.input.timedate,
            )
            self.data.prepare_output_inversion(cfg.data.output.N_emissions)
        else:
            logging.info("No model type selected.")
            sys.exit()

    def build_model(self, cfg: DictConfig) -> None:
        """Build the inversion or segmentation model."""

        if "load_weights" in cfg:
            self.model = tf.keras.models.load_model(
                os.path.join(cfg.dir_res, cfg.load_weights, "w_last.h5"),
                compile=True,
            )

        else:
            if cfg.model.type in ["inversion", "uq"]:
                try:
                    reg_builder = instantiate(
                        cfg.model.init,
                        input_shape=self.data.x.fields_input_shape,
                        classes=self.data.y.classes,
                        norm_layer=self.data.x.norm_layer,
                        noise_layer=self.data.x.noise_layer,
                        cloud_layer=self.data.x.cloud_layer,
                        timedate_vector_size=self.data.x.timedate_vector_size,
                    )

                    self.model = reg_builder.get_model()
                except Exception as e:
                    logging.error(f"An error occurred: {e}")
                    sys.exit()
            else:
                logging.info("No model type selected.")
                sys.exit()

    def compile_model(self, cfg: DictConfig) -> None:
        try:
            self.model.compile(
                optimizer=optimisers.define_optimiser(
                    cfg.training.optimiser, cfg.training.init_lr
                ),
                loss=loss.define_loss(cfg.training.loss_func),
                metrics=loss.define_metrics(cfg.model.type),
            )
        except Exception as e:
            logging.error(f"Compilation error: {e}")
            sys.exit()

    def prepare_training(self, cfg: DictConfig) -> None:
        """Prepare Trainer object."""
        generator = self.prepare_generator(cfg)
        callbacks = self.prepare_callbacks(cfg)

        try:
            self.trainer = Trainer(
                generator=generator,
                callbacks=callbacks,
                batch_size=cfg.training.batch_size,
                max_epochs=cfg.training.max_epochs,
            )

        except Exception as e:
            logging.error(f"An error occurred at Trainer preparation: {e}")
            sys.exit()

    def prepare_generator(self, cfg: DictConfig) -> generators.InvDataGen:
        """Prepare generator to feed to Trainer."""
        ic(cfg.model.type)
        if cfg.model.type == "inversion":
            try:
                generator = instantiate(
                    cfg.augmentations,
                    self.data.x.train,
                    self.data.x.xco2_plumes_train,
                    self.data.x.xco2_back_train,
                    self.data.x.xco2_alt_anthro_train,
                    getattr(self.data.x, "no2_plumes_train", None),
                    getattr(self.data.x, "no2_back_train", None),
                    self.data.y.train,
                    self.data.x.list_chans,
                    self.data.x.clouds_training,
                    self.data.x.noise_layer,
                    self.data.x.norm_layer,
                    self.data.x.cloud_layer,
                )
            except AttributeError as e:
                logging.error(f"Missing attribute at generator preparation: {e}")
                sys.exit()
            except Exception as e:
                logging.error(f"An error occurred at generator preparation: {e}")
                sys.exit()
        else:
            logging.info("No model type selected...")
            sys.exit()
        return generator

    def prepare_callbacks(self, cfg: DictConfig) -> list:
        """Prepare list of callbacks to feed to Trainer."""
        cbs = callbacks.get_modelcheckpoint(cfg.callbacks.model_checkpoint, [])
        cbs = callbacks.get_lrscheduler(cfg.callbacks.learning_rate_monitor, cbs)
        cbs = callbacks.get_wandb(cfg.callbacks.wandb, cbs)

        history_2 = callbacks.ExtraValidation(
            (self.data.x.valid_2, self.data.y.valid_2), "valid_2"
        )
        history = callbacks.ExtraValidation(
            (self.data.x.extra_valid, self.data.y.extra_valid), "extra_valid"
        )

        cbs.append(WandbMetricsLogger())
        cbs.append(history_2)
        cbs.append(history)
        return cbs

    def run(self) -> None:
        """Train the model with the training data."""
        try:
            self.model = self.trainer.train_model(self.model, self.data)
        except Exception as e:
            logging.error(f"An error occurred at RUN time: {e}")
            sys.exit()
        return self.trainer.get_val_loss()

    def save(self) -> None:
        """Save results of the run."""
        print("Saving at:", os.getcwd())
        self.saver.save_norm_layer(
            self.data.x.norm_layer, self.data.x.fields_input_shape
        )
        self.saver.save_cloud_layer(
            self.data.x.cloud_layer, self.data.x.fields_input_shape
        )
        self.saver.save_model_and_weights(self.model)
