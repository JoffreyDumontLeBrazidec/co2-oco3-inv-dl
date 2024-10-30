# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_dftp_hist
# TODO:
#
#

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pres_config import *
from matplotlib_functions import *

# download
df_pos = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)
title = "hour_hists_positive.pdf"

# prepare df_test
df_pos = df_pos.loc[df_pos["positivity"] == "positive"]
df_pos["tt_mean_quantile_1e2"] = pd.qcut(df_pos["tt_test_mean"] * 100, q=4, precision=2)
df_pos["tt_plume_var_quantile_1e2"] = pd.qcut(
    df_pos["tt_test_plume_var"] * 100, q=4, precision=1
)
df_pos["ratio_var_quantile_1e2"] = pd.qcut(df_pos["ratio_var"] * 100, q=4, precision=2)
df_pos["tt_plume_mean_quantile_1e2"] = pd.qcut(
    df_pos["tt_test_plume_mean"] * 100, q=4, precision=2
)
df_pos["tt_test_image_cover_quantile"] = pd.qcut(
    df_pos["tt_test_image_cover"], q=4, precision=2
)
df_pos["hour_quantile"] = pd.qcut(df_pos["hour"], q=4, precision=2)

df_pos.to_csv(os.path.join(dir_save_current_model, "list_infos_pos_predictions.csv"))

# prepare plot
variable_axs = [
    "tt_plume_mean_quantile_1e2",
    "tt_plume_var_quantile_1e2",
    "pred_success",
    "tt_test_image_cover_quantile",
]
xlabel_axs = ["", "", "Hour", "Hour"]
text_axs = ["(a)", "(b)", "(c)", "(d)"]
ylabel_axs = ["Ind. normed density", "", "Ind. normed density", ""]
output = "hour"
xlabels_ticks_size_axs = [6, 6, 6, 6]

# plot
setMatplotlibParam()
[ax1, ax2, ax3, ax4] = setFigure_2_2()
axs = [ax1, ax2, ax3, ax4]

for index_ax, ax in zip(range(len(axs)), axs):
    sns.histplot(
        data=df_pos,
        x=output,
        hue=variable_axs[index_ax],
        ax=ax,
        log_scale=False,
        element="step",
        cumulative=False,
        stat="density",
        common_norm=False,
    )
    # ax.set_ylim             (0.0,1)
    ax.text(
        0.05,
        0.94,
        text_axs[index_ax],
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
    )
    ax.set_xlabel(xlabel_axs[index_ax])
    ax.set_ylabel(ylabel_axs[index_ax])
    ax.tick_params(axis="x", which="major", labelsize=xlabels_ticks_size_axs[index_ax])
    set_sns_histplot_legend(
        ax, new_loc="lower left", prop={"size": 5}, title_fontsize=5
    )

axs = clean_axs_of_violin_plots(axs)

plt.savefig(os.path.join(dir_save_current_model, title), bbox_inches="tight")
