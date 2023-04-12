#!/usr/bin/env python

# list variables to exploit
def return_list_variables():
    
    list_all_variables = list()
    list_all_variables.append(
        [
            "orga.save.chain.bool",
            "True",
        ]
    )

    list_all_variables.append(
        [
            "model.callback.earlystopping",
            "False",
        ]
    )

    list_all_variables.append(
        [
            "model.name",
            "EfficientNetB0",
            "EfficientNetB2",
            "EfficientNetB4",            
            "ResNet50V2",
            "DenseNet121"
        ]
    )

    list_all_variables.append(
        [
            "model.init",
            "random",
            "imagenet"
        ]
    )
    
    list_all_variables.append(
        [
            "model.lr.base",
            "5E-3",
            "5E-4"
        ]
    )

    return list_all_variables

# __________________________________

def return_list_together_values_to_destroy():
    
    # combinations to destroy
    list_together_values_to_destroy = list()

    return list_together_values_to_destroy
