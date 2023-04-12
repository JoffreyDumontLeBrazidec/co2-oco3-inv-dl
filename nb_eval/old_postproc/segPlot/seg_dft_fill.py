# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of seg_dft_fill
# TODO:
#
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from seg_config import *
from matplotlib_functions import *
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
y_pred = PDM_model.predict(data.x_test)

# download df and fill it
df = pd.read_pickle(os.path.join(path_to_dataset, "df_infos_dataset.plk"))
df_test = df.loc[data.list_test_indices]
N_test = len(data.list_test_indices)

test_loss = np.empty(N_test)
bce = tf.keras.losses.BinaryCrossentropy()
for i in range(N_test):
    test_loss[i] = bce(y_test[i], y_pred[i]).numpy()
df_test["pred_loss"] = test_loss

tt_test_mean = tt_test.mean(axis=(1, 2))
df_test["tt_test_mean"] = tt_test_mean

tt_test_plume_mean = np.average(tt_test, axis=(1, 2), weights=tt_test.astype(bool))
df_test["tt_test_plume_mean"] = tt_test_plume_mean

tt_test_image_cover = (
    np.sum((tt_test > 1e-5), axis=(1, 2)) / (tt_test.shape[1] * tt_test.shape[2])
)[:, 0]
df_test["tt_test_image_cover"] = tt_test_image_cover

Z = np.array([tt_test[i] - tt_test_plume_mean[i] for i in range(tt_test.shape[0])])
tt_test_plume_var = np.average(Z ** 2, axis=(1, 2), weights=tt_test.astype(bool))
df_test["tt_test_plume_var"] = tt_test_plume_var

background_test_var = (f_test - tt_test).var(axis=(1, 2))
df_test["ratio_var"] = tt_test_plume_var / background_test_var

hour = [None] * len(df_test.index.values)
hour_block = [None] * len(df_test.index.values)
for index_test, norm_index_test in zip(
    df_test.index.values, range(len(df_test.index.values))
):
    hour[norm_index_test] = df_test.at[index_test, "time"] % 24
    hour_block[norm_index_test] = int(np.trunc(hour[norm_index_test] / 4) * 4)
df_test["hour"] = hour
df_test["hour_block"] = hour_block

df_test.to_csv(os.path.join(dir_save_current_model, "list_infos_predictions.csv"))

# __________________________________________________________
