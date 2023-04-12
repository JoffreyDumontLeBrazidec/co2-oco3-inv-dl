# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
from include.callbacks import initiate_wb
from model_training import Model_training_manager

dir_res = "/cerea_raid/users/dumontj/dev/coco2/dl/res"
name_case = "debug"
cfg = OmegaConf.load(os.path.join(os.path.join(dir_res, name_case), "config.yaml"))
print(OmegaConf.to_yaml(cfg, resolve=True))
initiate_wb(cfg)
model_trainer = Model_training_manager(cfg)
model_trainer.run()
model_trainer.save()


