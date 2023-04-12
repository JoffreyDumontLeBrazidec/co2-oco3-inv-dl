# ----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
# ----------------------------------------------------------------------------

import os
import numpy as np
import sys
import pandas as pd
import xarray as xr
import joblib
import pickle

import tensorflow as tf
sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl")
from include.loss import pixel_weighted_cross_entropy

import scipy.ndimage
import scipy.stats
import shapely



def shift_to_proba(y_pred, proba_max, proba_min):
    """Shift to a probability map from a boolean map: 1 to proba_max, 0 to proba_min."""
    y_pred = np.where(y_pred == 1, proba_max, proba_min)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    return y_pred

def get_all_loss(y_test, pred_test):
    """Get all losses."""
    all_loss_test = pixel_weighted_cross_entropy(y_test, pred_test, reduction=False)
    all_loss_test = np.mean(all_loss_test, axis=(1,2))
    return all_loss_test

def get_mean_loss(params, y_test, pred_test):
    """Get mean wbce between y_test and pred_test given shift_to_proba with params."""
    proba_min, proba_max = params
    current_pred_test = shift_to_proba(pred_test,proba_max,proba_min)
    all_loss_test = get_all_loss(y_test, current_pred_test)
    return np.mean(all_loss_test)