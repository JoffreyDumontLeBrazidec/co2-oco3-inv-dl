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


def check_diff_values(list_of_dicts):
    from collections import defaultdict

    value_tracker = defaultdict(set)

    def add_values(d, parent_key=""):
        for k, v in d.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                add_values(v, full_key)
            else:
                value_tracker[full_key].add(v)

    for d in list_of_dicts:
        add_values(d)

    keys_with_diff_values = [
        key for key, values in value_tracker.items() if len(values) > 1
    ]
    return keys_with_diff_values


def filter_name_models(list_names_model, conditions, dir_sweep):
    """
    Remove dictionaries from a list where specific keys do not match given values.

    :param list_of_dicts: List of dictionaries to filter.
    :param conditions: Dictionary of key-value pairs that must be met to keep a dictionary.
                       Keys support nested notation with '.', e.g., 'data.init._target_'.
    :return: Filtered list of dictionaries.
    """

    def match_conditions(d, conditions, parent_key=""):
        for key, expected_value in conditions.items():
            keys = key.split(".")
            current_d = d
            try:
                for k in keys:
                    if isinstance(current_d, dict):
                        current_d = current_d[k]
                    else:
                        return False
                if current_d != expected_value:
                    return False
            except KeyError:
                return False
        return True

    filtered_list_names_models = []
    for name_model in list_names_model:
        config_path = os.path.join(dir_sweep, name_model, "config.yaml")
        config = OmegaConf.load(config_path)
        config_dict = OmegaConf.to_container(config, resolve=True)
        if match_conditions(config_dict, conditions):
            filtered_list_names_models.append(name_model)

    return filtered_list_names_models
