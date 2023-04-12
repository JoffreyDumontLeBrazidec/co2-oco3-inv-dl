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
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import joblib
import pickle
from treeconfigparser import TreeConfigParser

import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from matplotlib import colors
from matplotlib import ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cmcrameri import cm
import matplotlib_functions
matplotlib_functions.setMatplotlibParam()
plt.viridis()


import scipy.ndimage
import scipy.stats
import shapely
import skimage.measure
import xarray
import dplume
import ddeq

import build_ds
import ddeq_pred

ddeq_pred.get_ddeq_predictions_forall_PS()
