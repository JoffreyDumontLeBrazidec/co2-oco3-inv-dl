# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_dft_fill
# TODO:
#
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow import keras
import xarray as xr
import os

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl")
from config import *
from data.Data import Data

# download data
shuffleIndices = np.array(
    np.fromfile(os.path.join(dir_save_current_model, "shuffle_indices.bin"))
).astype("int")

data = Data(config)
data.prepareXCO2Data(shuffleIndices=shuffleIndices)
data.download_tt_dataset()

f_test = data.x_test[0]
y_test = data.y_test
tt_test = data.tt_test
tt_test[:, 0, 0] = -0.01  # avoid having only zero values

# download model and make predictions
PDM_model = keras.models.load_model(os.path.join(dir_save_current_model, "PDM.h5"))
if PDM_model.layers[0].get_output_at(0).get_shape()[-1] == 3:
    data.superpose_field_in_three_channels()
test_predictions = PDM_model.predict(data.x_test)

# download df and fill it
ds = xr.open_dataset(path_to_dataset)
ds_test = ds.sel(index_image=data.list_test_indices)

data = {
    "index_image": ds_test.index_image.values,
    "folder": ds_test.folder.values,
    "positivity": ds_test.ppresence.values,
    "cropping": (
        (ds_test.shape_cropping[:, 3].values - ds_test.shape_cropping[:, 2].values)
        * (ds_test.shape_cropping[:, 1].values - ds_test.shape_cropping[:, 0].values)
    )
    / (Ny * Nx),
    "time": ds_test.time.values,
    "pred_pos_perc": np.squeeze(test_predictions),
    "pred_success": (np.squeeze(np.around(test_predictions)) == y_test),
    "plume_mean": np.squeeze(np.average(tt_test, axis=(1, 2), weights=tt_test.astype(bool))),
    "plume_img_cover": np.squeeze(
        np.sum((tt_test > 1e-5), axis=(1, 2)) / (tt_test.shape[1] * tt_test.shape[2])
    ),
    "plume_var": np.squeeze(np.average(
        (
            tt_test
            - (np.average(tt_test, axis=(1, 2), weights=tt_test.astype(bool))).reshape(
                -1, 1, 1, 1
            )
        )
        ** 2,
        axis=(1, 2),
        weights=tt_test.astype(bool),
    )),
    
    "dist_to_truth": abs(np.squeeze(test_predictions) - y_test),

    "hour": (ds_test.time % 24).values,
    "hour_block": (np.trunc((ds_test.time % 24) / 4) * 4).values.astype(np.int64),
}

# Create DataFrame
df_test = pd.DataFrame(data)

df_test.to_csv(os.path.join(dir_save_current_model, "list_infos_predictions.csv"))

# __________________________________________________________
