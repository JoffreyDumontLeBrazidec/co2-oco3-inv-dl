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
            "model.name",
            "multiFields_CNN",
        ]
    )

    list_all_variables.append(
        [
            "model.optimiser",
            "adam",
            "rmsprop",
            "adadelta"
        ]
    )

    list_all_variables.append(
        [
            "model.learning_rate",
            "1e-5",
            "1e-4",
            "5e-4",
            "1e-3",
            "5e-3",
            "1e-2",
            "5e-2"
        ]
    )

    list_all_variables.append(["model.epochs.number", "250"])

    list_all_variables.append(
        [
            "model.batch_size",
            "64",
        ]
    )

    list_all_variables.append(
        ["data.directory.name", "CO2_pPGI_nBB_pure"]
    )

    list_all_variables.append(
        [
            "data.input.time",
            "False",
        ]
    )

    list_all_variables.append(
        [
            "data.input.scale",
            "False",
        ]
    )

    list_all_variables.append(
        [
            "data.input.winds.format",
            "none",
        ]
    )


    list_all_variables.append(
        [
            "data.input.dynamics.format",
            "none",
        ]
    )

    return list_all_variables

# __________________________________

def return_list_together_values_to_destroy():
    
    # combinations to destroy
    list_together_values_to_destroy = list()

    return list_together_values_to_destroy
