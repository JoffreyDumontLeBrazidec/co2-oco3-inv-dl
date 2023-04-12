# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import numpy as np

from omegaconf import DictConfig, OmegaConf
import hydra

from model_training import Model_training_manager

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main_train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    try:
        model_trainer = Model_training_manager(cfg)
        model_trainer.run()
        model_trainer.save()
        return model_trainer
    except MemoryError:
        print("Memory error issue")
        sys.exit()

if __name__ == '__main__':
    model_trainer = main_train()



