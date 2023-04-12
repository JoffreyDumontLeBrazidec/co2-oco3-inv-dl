# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_dft_hist
# TODO:
#
#

import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_functions import *
from config import *

# download
df_test = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)

# prepare df_test
df_test["cropping_block"] = pd.qcut(
    df_test["cropping"]*100, q=[0, 0.2, 0.4, 0.6, 0.8, 1], precision=1
)

# prepare plot
variable_axs = ["folder", "cropping_block", "hour_block", "positivity"]
xlabel_axs = ["", "", "distance to truth", "distance to truth"]
text_axs = ["(a)", "(b)", "(c)", "(d)"]
ylabel_axs = ["Ind. normed density", "", "Ind. normed density", ""]
output = "dist_to_truth"
xlabels_ticks_size_axs = [6, 6, 6, 6]

# plot
setMatplotlibParam()
[ax1, ax2, ax3, ax4] = setFigure_2_2()
axs = [ax1, ax2, ax3, ax4]

for index_ax, ax in zip(range(len(axs)), axs):
    sns.histplot(
        data=df_test,
        x=output,
        hue=variable_axs[index_ax],
        ax=ax,
        log_scale=False,
        element="step",
        cumulative=True,
        stat="density",
        common_norm=False,
    )
    ax.set_ylim(0.6, 1)
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
    ax.axvline(x=0.5, ls="--", lw=1.0, c="black")

plt.savefig(os.path.join(dir_save_current_model, "dtt_hists_full.pdf"), bbox_inches="tight")
