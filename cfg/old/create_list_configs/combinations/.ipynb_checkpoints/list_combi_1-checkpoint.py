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
            "multiFields_vector_CNN",
        ]
    )

    list_all_variables.append(
        [
            "model.optimiser",
            "adam",
        ]
    )

    list_all_variables.append(
        [
            "model.learning_rate",
            "1e-3",
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
        ["data.directory.name", "CO2_pPGI_nBB", "CO2_pPGI_nBBO", "CO2_pPG_nBBOI"]
    )

    list_all_variables.append(
        [
            "data.input.time",
            "False",
            "True",
        ]
    )

    list_all_variables.append(
        [
            "data.input.scale",
            "False",
            "True",
        ]
    )

    list_all_variables.append(
        [
            "data.input.winds.format",
            "none",
            "summary",
            "fields",
        ]
    )

    list_all_variables.append(
        [
            "data.input.winds.fields.number.current",
            "2",
            "6",
        ]
    )

    list_all_variables.append(
        [
            "data.input.dynamics.format",
            "none",
            "fields",
        ]
    )

    list_all_variables.append(
        [
            "data.input.dynamics.fields.number.current",
            "1",
            "3",
        ]
    )

    return list_all_variables

def return_list_together_values_to_destroy():
    
    # combinations to destroy
    list_together_values_to_destroy = list()

    list_together_values_to_destroy.append(
        [["model.name", "multiFields_CNN"], ["data.input.time", "True"]]
    )
    list_together_values_to_destroy.append(
        [["model.name", "multiFields_CNN"], ["data.input.scale", "True"]]
    )
    list_together_values_to_destroy.append(
        [["model.name", "multiFields_CNN"], ["data.input.winds.format", "summary"]]
    )

    list_together_values_to_destroy.append(
        [
            ["model.name", "multiFields_vector_CNN"],
            ["data.input.winds.format", "fields"],
            ["data.input.time", "False"],
        ]
    )
    list_together_values_to_destroy.append(
        [
            ["model.name", "multiFields_vector_CNN"],
            ["data.input.winds.format", "none"],
            ["data.input.time", "False"],
        ]
    )

    list_together_values_to_destroy.append(
        [
            ["data.input.winds.format", "none"],
            ["data.input.winds.fields.number.current", "6"],
        ]
    )
    list_together_values_to_destroy.append(
        [
            ["data.input.dynamics.format", "none"],
            ["data.input.dynamics.fields.number.current", "3"],
        ]
    )

    return list_together_values_to_destroy
