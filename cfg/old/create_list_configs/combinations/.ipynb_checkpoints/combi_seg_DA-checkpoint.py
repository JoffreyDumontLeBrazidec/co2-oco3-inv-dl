def return_dic_hyperparameters():
    """Create and return dictionnary of keys-values to combine."""

    base_cfg = "seg_PGIPW_DA.cfg"
    
    dic_hyperparameters = {"orga.save.chain.bool": ["True"],
                           "data.input.xco2.noise.level": ["0.7"],
                           "model.name": ["Unet_4", "Unet_5", "Unet_efficient"],
                          }

    return [base_cfg, dic_hyperparameters]