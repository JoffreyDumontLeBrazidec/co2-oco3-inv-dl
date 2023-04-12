#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
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

from matplotlib_functions import *
from config import *

# download
df_test = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)

# prepare df_test
df_pos = df_test.loc[df_test['positivity']]
df_pos["plume_mean_1e2"]          = pd.qcut(df_pos["plume_mean"]*100, q=4, precision=2)
df_pos["plume_var_1e2"]     = pd.qcut(df_pos["plume_var"]*100, q=4, precision=1)
df_pos["plume_image_cover"]  = pd.qcut(df_pos["plume_img_cover"], q=4, precision=2)
df_pos["hour_block"]                 = pd.qcut(df_pos["hour"], q=4, precision=2)

# prepare plot
variable_axs    = ["plume_mean_1e2", "plume_var_1e2", "hour_block", "plume_image_cover"]
xlabel_axs      = ["", "", "distance to truth", "distance to truth"]
text_axs        = ["(a)", "(b)", "(c)", "(d)"]
ylabel_axs      = ["Ind. normed density", "", "Ind. normed density", ""]
output          = "dist_to_truth"
xlabels_ticks_size_axs    = [6,6,6,6]

# plot
setMatplotlibParam()
[ax1, ax2, ax3, ax4] = setFigure_2_2()
axs = [ax1, ax2, ax3, ax4]

for index_ax, ax in zip (range(len(axs)), axs):
    sns.histplot            (data=df_pos, x=output, hue=variable_axs[index_ax], ax=ax, log_scale=False, element="step", cumulative=True, stat="density", common_norm=False, bins=50)
    ax.set_ylim             (0.6,1)
    ax.text                 (0.05, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel           (xlabel_axs[index_ax])
    ax.set_ylabel           (ylabel_axs[index_ax])
    ax.tick_params          (axis='x', which='major', labelsize=xlabels_ticks_size_axs[index_ax])
    set_sns_histplot_legend (ax, new_loc="lower left", prop = {"size" : 5}, title_fontsize=5)
    ax.axvline              (x=0.5, ls ='--', lw = 1., c = 'black')

plt.savefig(os.path.join(dir_save_current_model, "dtt_hists_positive.pdf"), bbox_inches="tight")










