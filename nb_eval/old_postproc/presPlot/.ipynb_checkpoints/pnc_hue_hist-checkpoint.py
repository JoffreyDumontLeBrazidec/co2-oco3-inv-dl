#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of pres_dftpnc_hist
# TODO: 
#
#

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from pres_config                import *
from local_importeur            import *

from tools.df_plots     import cumulative_histograms_with_hue, shape_continuous_value_in_block_for_df
from tools.tools_postproc               import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_color_map, clean_axs_of_violin_plots, set_sns_histplot_legend

# download
df_nc_pos = pd.read_csv (dir_save_current_model + "/" + "list_infos_predictions.csv", index_col=0)
title = "dtt_hists_noncropped_positive.pdf"

# prepare df_test
df_nc_pos = df_nc_pos.loc[(df_nc_pos['positivity'] == "positive") & (df_nc_pos['cropping'] == False)]
df_nc_pos["tt_mean_quantile_1e2"]          = pd.qcut(df_nc_pos["tt_test_mean"]*100, q=4, precision=2)
df_nc_pos["tt_plume_var_quantile_1e2"]     = pd.qcut(df_nc_pos["tt_test_plume_var"]*100, q=4, precision=1)
df_nc_pos["ratio_var_quantile_1e2"]        = pd.qcut(df_nc_pos["ratio_var"]*100, q=4, precision=2)
df_nc_pos["tt_plume_mean_quantile_1e2"]    = pd.qcut(df_nc_pos["tt_test_plume_mean"]*100, q=4, precision=2)
df_nc_pos["tt_test_image_cover_quantile"]  = pd.qcut(df_nc_pos["tt_test_image_cover"], q=4, precision=2)
df_nc_pos["hour_quantile"]                 = pd.qcut(df_nc_pos["hour"], q=[0, .5, .58, .7, .8, 1], precision=2)

# prepare plot
variable_axs            = ["tt_plume_mean_quantile_1e2", "tt_plume_var_quantile_1e2", "hour_quantile", "tt_test_image_cover_quantile"]
xlabel_axs              = ["", "", "distance to truth", "distance to truth"]
text_axs                = ["(a)", "(b)", "(c)", "(d)"]
ylabel_axs              = ["Ind. normed density", "", "Ind. normed density", ""]
output                  = "distance_to_truth"
xlabels_ticks_size_axs  = [6,6,6,6]

# plot    
setMatplotlibParam()
[ax1, ax2, ax3, ax4] = setFigure_2_2()
axs = [ax1, ax2, ax3, ax4]

for index_ax, ax in zip (range(len(axs)), axs):
    #df_nc_pos.pivot         (columns=variable_axs[index_ax], values='distance_to_truth').plot.hist(cumulative=True, density=1, bins=100,ax=ax,alpha=0.33, histtype='stepfilled')
    df_nc_pos.pivot         (columns=variable_axs[index_ax], values='distance_to_truth').plot.hist(cumulative=True, density=1, bins=200,ax=ax,alpha=1, histtype="step")
    ax.set_ylim             (0.6,1)
    ax.text                 (0.05, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    ax.set_xlabel           (xlabel_axs[index_ax])
    ax.set_ylabel           (ylabel_axs[index_ax])
    ax.tick_params          (axis='x', which='major', labelsize=xlabels_ticks_size_axs[index_ax])
    set_sns_histplot_legend (ax, new_loc="lower right", prop = {"size" : 5}, title_fontsize=5)
    ax.axvline              (x=0.5, ls ='--', lw = 1., c = 'black')

axs = clean_axs_of_violin_plots (axs)
    
plt.savefig (dir_save_current_model + '/' + title, bbox_inches= 'tight')











