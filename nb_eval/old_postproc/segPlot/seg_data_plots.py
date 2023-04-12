# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
## Implementation of segmentation_plots
# TODO:
#
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow import keras
import shutil

from seg_config import *
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

# download df_test
df_test = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)
index_in_df_test = np.arange(0, len(df_test))
df_test["index_in_df_test"] = index_in_df_test

# plot mix-fields
dir_plot_fields = os.path.join(dir_save_current_model, "images")
if os.path.exists(dir_plot_fields):
    shutil.rmtree(dir_plot_fields)
os.makedirs(dir_plot_fields)

number_random_choices = 1
for index_prediction in np.random.choice(list(df_test.index), number_random_choices):

    # prepare plot
    index_prediction_in_test = df_test.loc[index_prediction, "index_in_df_test"]
    current_f_test = f_test[index_prediction_in_test, :, :, 0]
    current_tt_test = tt_test[index_prediction_in_test, :, :, 0]
    current_y_test = y_test[index_prediction_in_test, :, :, 0]
    current_y_pred = y_pred[index_prediction_in_test, :, :, 0]

    title_plot = (
        "Truth:"
        + str(df_test.loc[index_prediction, "positivity"])
        + "; bce:"
        + "%.3f" % df_test.loc[index_prediction, "pred_loss"]
        + "; cropping: "
        + str(df_test.loc[index_prediction, "cropping"])
        + "; t: "
        + str(df_test.loc[index_prediction, "time"] % 24)
        + "h"
    )

    title_save = "test_image_" + str(index_prediction) + ".png"

    # plot
    setMatplotlibParam()
    [ax1, ax2, ax3, ax4] = setFigure_2_2()
    axs = [ax1, ax2, ax3, ax4]

    c1 = ax1.pcolor(
        current_f_test.transpose(1, 0),
        cmap=download_color_map(),
        edgecolor="face",
        zorder=0,
    )
    c2 = ax2.pcolor(
        current_tt_test.transpose(1, 0), cmap="viridis", alpha=0.35, zorder=0
    )
    c3 = ax3.pcolor(
        current_y_test.transpose(1, 0), cmap="viridis", alpha=0.35, zorder=0
    )
    c4 = ax4.pcolor(
        current_y_pred.transpose(1, 0), cmap="viridis", alpha=0.35, zorder=0
    )

    for ax in axs:
        ax.grid(False)

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c4, cax, orientation="vertical")
    cbar.ax.tick_params(labelsize="7")

    # ax_title = fig.add_subplot(212, frameon = False)
    ax3.set_title(title_plot, fontsize=7, loc="right")
    plt.savefig(os.path.join(dir_plot_fields, title_save))
