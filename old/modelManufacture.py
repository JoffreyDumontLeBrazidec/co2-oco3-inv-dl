#----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import datetime

from data.Data import Data
from include.generators import Generator
from include.callbacks import create_list_callbacks
from include.optimisers import define_optimiser
from include.loss import loss
import models.seg as sm
import models.pw_reg as pwrm
import models.reg as rm
import models.pres as pm

# __________________________________________________________
# PlumeDetectionMachine
class PlumeDetectionMachine:

    # ----------------------------------------------------
    # __initialiser__
    def __init__(self, config, Data, **kwargs):

        # data
        self.data = Data

        # model 
        ## definition
        self.model_purpose = config.get("model.purpose")
        self.model_input = config.get("model.input")
        self.model_name = config.get("model.name")
        
        ## parameters
        self.model_init = config.get("model.init")
        self.batch_size = config.get_int("model.batch_size")
        self.N_epochs = config.get_int("model.epochs.number")
        self.N_steps_per_epoch = int(np.floor(self.data.N_train / self.batch_size))

        ## include
        self.loss = define_loss(config)
        self.generator = Generator(config)
        self.opt = define_optimiser(config)
        self.list_callbacks = create_list_callbacks(config)
        
        # general settings
        policy = tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # ------------------------------------------------------------------------
    # buildModel
    def buildModel(self):
        
        self.model = None
        
        # presence
        if self.model_purpose == "presence":
            ## standard
            if self.model_name.startswith("standard_cnn"):
                ### with field
                if self.model_input == "field":
                    self.model = pm.build_multiFields_CNN(
                        input_shape=self.data.fields_input_shape,
                        out_Nclasses=self.data.classes,
                    )
                ### with field and vector
                elif self.model_input == "field_vector":
                    self.model = pm.build_multiFields_vector_CNN(
                        fields_input_shape=self.data.fields_input_shape,
                        vector_input_shape=self.data.vector_input_shape,
                        out_Nclasses=self.data.classes,
                    )
                else:
                    print("Presence standard model input poorly defined")
                    sys.exit()
            ## classic
            elif (
                self.model_name.startswith("EfficientNet")
                or self.model_name.startswith("ResNet")
                or self.model_name.startswith("DenseNet")
            ):  
                ### with field
                if self.model_input == "field":
                    self.data.superpose_field_in_three_channels()
                    self.model = pm.build_multiFields_classicNN(
                        input_shape=self.data.fields_input_shape,
                        name=self.model_name,
                        init_weights=self.model_init,
                        out_Nclasses=self.data.classes,
                    )
                else:
                    print("Presence classic model input poorly defined")
            else:
                print("Presence model name not defined")
                sys.exit()
                
            self.model.compile(
                optimizer=self.opt,
                loss=self.loss,
                metrics=["accuracy"],
            )
        # segmentation
        elif self.model_purpose == "segmentation":
            ## U-net
            if self.model_name == "Unet":
                if self.model_input == "field":
                    self.model = sm.build_Unet_2(
                        out_Nclasses=self.data.classes,
                        in_Ny=self.data.Ny,
                        in_Nx=self.data.Nx,
                        in_Nchannels=1,
                    )
                else:
                    print("Segmentation Unet model input poorly defined")
            else:
                print("Segmentation model name poorly defined")
                
            self.model.compile(optimizer=self.opt, loss=self.loss, metrics=[])
        else:
            print("Model purpose poorly defined")
        
        if self.model is None:
            print("No model defined")
            sys.exit()
            
        self.model.summary()

    # ------------------------------------------------------------------------
    # train
    def train(self):

        self.history = self.model.fit(
            self.generator.flow_on_data(self.data.x_train, self.data.y_train),
            epochs=self.N_epochs,
            steps_per_epoch=self.N_steps_per_epoch,
            validation_data=(self.data.x_valid, self.data.y_valid),
            verbose=1,
            callbacks=self.list_callbacks,
        )

        self.test_metrics = self.model.evaluate(
            x=self.data.x_test, y=self.data.y_test, batch_size=32, verbose=1
        )

# __________________________________________________________
