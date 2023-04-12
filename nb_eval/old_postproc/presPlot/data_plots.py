# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_data_plots
# TODO:
#
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import shutil

from config import *
from tools.data_plots import plot_field
sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl")
from data.Data import Data

#------------------------------------------------------------------------
# import_data
def import_data(config, shuffleIndices):

    data                           = Data(config)
    data.prepareXCO2Data           (shuffleIndices = shuffleIndices)
    data.download_tt_dataset       ()
    return data

#------------------------------------------------------------------------

# download data
shuffleIndices = np.array(
    np.fromfile(os.path.join(dir_save_current_model, "shuffle_indices.bin"))
).astype("int")

data = Data(config)
data.prepareXCO2Data(shuffleIndices = shuffleIndices)
data.download_tt_dataset()

list_test_indices_considered = list(
    shuffleIndices[data.N_trainingData + data.N_validationData :]
)
f_test = data.x_test[0]
y_test = data.y_test
tt_test = data.tt_test
tt_test[:, 0, 0] = -0.01  # avoid having only zero values

# download df_test
df_test = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)
index_in_df_test = np.arange(0, len(df_test))
df_test["index_in_df_test"] = index_in_df_test

# plot wrong predictions
dir_plot_fields_wrongly_predicted = os.path.join(
    dir_save_current_model, "images_wrong_predictions"
)
if os.path.exists(dir_plot_fields_wrongly_predicted):
    shutil.rmtree(dir_plot_fields_wrongly_predicted)
os.makedirs(dir_plot_fields_wrongly_predicted)
df_wrong_predictions = df_test.loc[df_test["pred_success"] == False]

number_random_choices = 30
for index_wrong_prediction in np.random.choice(
    list(df_wrong_predictions.index), number_random_choices
):

    index_wrong_prediction_in_test = df_wrong_predictions.loc[
        index_wrong_prediction, "index_in_df_test"
    ]
    current_f_test = f_test[index_wrong_prediction_in_test, :, :, 0]
    current_tt_test = tt_test[index_wrong_prediction_in_test, :, :, 0]
    title_plot = "".join(
        [
            "Truth:",
            str(df_wrong_predictions.loc[index_wrong_prediction, "positivity"]),
            "; dtt:",
            "%.3f"
            % df_wrong_predictions.loc[index_wrong_prediction, "distance_to_truth"],
            "; cropping: ",
            str(df_wrong_predictions.loc[index_wrong_prediction, "cropping"]),
            "; t: ",
            str(df_wrong_predictions.loc[index_wrong_prediction, "time"] % 24),
            "h",
        ]
    )

    plot_field(
        dir_plot_fields_wrongly_predicted,
        current_f_test,
        current_tt_test,
        df_wrong_predictions.loc[index_wrong_prediction],
        index_wrong_prediction,
        title=title_plot,
    )

df_wrong_predictions.to_csv(
    os.path.join(dir_save_current_model, "list_infos_wrong_predictions.csv")
)

# plot right predictions
dir_plot_fields_rightly_predicted = os.path.join(
    dir_save_current_model, "images_right_predictions"
)
if dir_plot_fields_rightly_predicted:
    shutil.rmtree(dir_plot_fields_rightly_predicted)
os.makedirs(dir_plot_fields_rightly_predicted)
df_right_predictions = df_test.loc[df_test["pred_success"] == True]

number_random_choices = 30
for index_right_prediction in np.random.choice(
    list(df_right_predictions.index), number_random_choices
):

    index_right_prediction_in_test = df_right_predictions.loc[
        index_right_prediction, "index_in_df_test"
    ]
    current_f_test = f_test[index_right_prediction_in_test, :, :, 0]
    current_tt_test = tt_test[index_right_prediction_in_test, :, :, 0]
    title_plot = "".join(
        [
            "Truth:",
            str(df_wrong_predictions.loc[index_wrong_prediction, "positivity"]),
            "; dtt:",
            "%.3f"
            % df_wrong_predictions.loc[index_wrong_prediction, "distance_to_truth"],
            "; cropping: ",
            str(df_wrong_predictions.loc[index_wrong_prediction, "cropping"]),
            "; t: ",
            str(df_wrong_predictions.loc[index_wrong_prediction, "time"] % 24),
            "h",
        ]
    )

    plot_field(
        dir_plot_fields_rightly_predicted,
        current_f_test,
        current_tt_test,
        df_right_predictions.loc[index_right_prediction],
        index_right_prediction,
        title=title_plot,
    )
