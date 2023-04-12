def return_dic_hyperparameters():
    """Create and return dictionnary of keys-values to combine."""

    base_cfg = "/cerea_raid/users/dumontj/dev/coco2/dl/cfg/pres.cfg"
    
    dic_hyperparameters = {"orga.save.chain.bool": ["True"],
                          "model.lr.max": ["1E-1"],
                          "model.lr.min": ["1E-4", "1E-5", "1E-6", "1E-7"],
                          "model.lr.decay.power": ["0.5", "1"]
                        }

    return [base_cfg, dic_hyperparameters]