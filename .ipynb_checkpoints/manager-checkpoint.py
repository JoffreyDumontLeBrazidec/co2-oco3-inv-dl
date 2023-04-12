#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import shutil
import os
import time
from typing import Any
from treeconfigparser import TreeConfigParser
import tensorflow as tf 
from dataclasses import dataclass

from include.generators import Generator
from include.callbacks import create_list_callbacks
from include.optimisers import define_optimiser
from include.loss import loss
import models.seg as sm
import models.pw_reg as pwrm
import models.reg as rm
import models.pres as pm
from data.Data import Data
from saver import Saver


@dataclass
class Builder:
    """Build CNN models."""
        model_purpose: str
        
    def build_model(self, data, name, init_w, input_shape, classes):
        """Build pres, seg, inv, or pw_inv model."""

        if self.model_purpose == "presence":
            # A faire comme Regression

        elif self.model_purpose == "segmentation":
            self.model = sm.build_Unet_2(input_shape, classes)
            
        elif self.model_purpose == "regression":
            reg_builder = rm.Reg_model_builder(name, input_shape, classes, init_w)
            self.model = reg_builder.get_model()
            
        elif self.model_purpose = "pixel_wise_regression":
            self.model = pwrm.build_Unet_2(input_shape, classes)
            
        return model
    

@dataclass
class Trainer:
    """Train CNN models."""
    loss: Any
    generator: Generator
    opt: Any
    callbacks: list
    batch_size: int
    N_epochs: int

    def __post_init__(self):
        self.N_steps_per_epoch = int(np.floor(self.data.N_train / self.batch_size))
    
    def compile_model(self, model):
        """Compile model with optimiser and loss function."""
        model.compile(optimizer=self.opt, loss=self.loss)
        return model
        
    def train_model(self, model, data):
        """Train model and evaluate validation and test metrics."""
        self.history = model.fit(
            self.generator.flow_on_data(data.x.train, data.y.train),
            epochs=self.N_epochs,
            validation_data=(data.x.valid, data.y.valid),
            verbose=1,
            callbacks=self.list_callbacks,
        )
        #steps_per_epoch=self.N_steps_per_epoch,

        self.test_metrics = model.evaluate(
            x=data.x_test, y=data.y_test, batch_size=32, verbose=1
        )        
        return model

@dataclass
class Model_training_manager:
    """Train CNN models for presence, segmentation, inversion, pixel-wise inversion tasks."""
    config_file: str
    
    def __post_init__(self):
        config = TreeConfigParser()
        config.readfiles(self.config_file)

        self.prepare_exp_dir(config)
        
        self.builder = Builder(config.get("model.purpose"))        
        self.builder.build_model(config.get("model.name"), config.get("model.init"), self.data.fields_input_shape, self.data.classes)
        
        self.trainer = Trainer(define_loss(config), Generator(config), define_optimiser(config), create_list_callbacks(config), config.get_int("model.batch_size"), config.get_int("model.epochs.number"))
        
        self.data = Data(config)
        self.data.prepare_input()
        self.data.prepare_output()

        self.saver = Saver(config, self.config_file)
        
    # ------------------------------------------------------------------------
    # run
    def run(self):

        # build, train, evaluate, save model
        training_timeStart = time.time()
        self.pdm.buildModel()
        self.pdm.train()
        training_time = time.time() - training_timeStart

        # saving results
        self.saver.save_training_time(training_time)
        self.saver.save_model(self.pdm.model)
        self.saver.save_history(self.pdm.history, self.pdm.test_metrics)
        self.saver.save_data_shuffle_indices(self.data.shuffleIndices, [self.data.train_indices, self.data.valid_indices, self.data.test_indices])

# __________________________________________________________


