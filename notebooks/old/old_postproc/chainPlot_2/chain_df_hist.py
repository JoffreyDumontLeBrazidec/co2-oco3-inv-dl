#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/chainPlot/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of chain_df_violin
# TODO: 
#
#

import sys
from matplotlib_functions import *
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from chain_config                import *
from local_importeur            import *


# download data
df = pd.read_csv (dir_all_models + "/" + "filled_table_combinations.csv")
title = "val_accuracy_hists.pdf"

# prepare df
df["N_epochs"] = df["N_epochs"].astype(int)
lower, higher = df['N_epochs'].min(), df['N_epochs'].max()
n_bins = 7
edges = range(lower, higher, int((higher - lower)/n_bins)) 
lbs = ['(%d, %d]'%(edges[i], edges[i+1]) for i in range(len(edges)-1)]
df["N_epochs_quantile"] = pd.cut(df.N_epochs, bins=n_bins, labels=lbs, include_lowest=True)
df["dataset"] = df["dataset"].str[4:]

# prepare plot
variable_axs    = ["input.time", "wind_as_input", "dynamic_as_input", "ata.input.scale", "dataset", "N_epochs_quantile"]
xlabel_axs      = ["", "", "", "", "test accuracy (best)", "test accuracy (best)"]
text_axs        = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
ylabel_axs      = ["Density", "", "Density", "", "Density", ""]
output          = "test_accuracy.best"
xlabels_ticks_size_axs  = [6,6,6,6,6,6]

# plot
setMatplotlibParam()
[ax1, ax2, ax3, ax4, ax5, ax6] = setFigure_2_2_2()
axs = [ax1, ax2, ax3, ax4, ax5, ax6]

for index_ax, ax in zip (range(len(axs)), axs):
    sns.histplot            (data=df, x=output, hue=variable_axs[index_ax], ax=ax, log_scale=False, element="step", cumulative=False, stat="density", common_norm=False)
    #ax.set_ylim             (0.6,1)
    ax.text                 (0.95, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel           (xlabel_axs[index_ax])
    ax.set_ylabel           (ylabel_axs[index_ax])
    ax.tick_params          (axis='x', which='major', labelsize=xlabels_ticks_size_axs[index_ax])
    set_sns_histplot_legend (ax, new_loc="upper left", prop = {"size" : 5}, title_fontsize=5)

plt.savefig (dir_all_models + "/" + title, bbox_inches= 'tight')
print ("PLOT:", dir_all_models + "/" + title)

