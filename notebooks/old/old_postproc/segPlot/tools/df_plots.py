#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of df_plots
# TODO: 
#
#

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from local_importeur                    import *

from tools.tools_postproc               import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_color_map, clean_axs_of_violin_plots, set_sns_histplot_legend

#------------------------------------------------------------------------
# cumulative_histograms_with_hue
def cumulative_histograms_with_hue (dir_save, df, output, variable_axs, xlabel_axs, text_axs, ylabel_axs, xlabels_ticks_size_axs, title):

    
    
    setMatplotlibParam()
    [ax1, ax2, ax3, ax4] = setFigure_2_2()
    axs = [ax1, ax2, ax3, ax4]

    for index_ax, ax in zip (range(len(axs)), axs):
        sns.histplot            (data=df, x=output, hue=variable_axs[index_ax], ax=ax, log_scale=False, element="step", cumulative=True, stat="density", common_norm=False)
        ax.set_ylim             (0.6,1)
        ax.text                 (0.05, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xlabel           (xlabel_axs[index_ax])
        ax.set_ylabel           (ylabel_axs[index_ax])
        ax.tick_params          (axis='x', which='major', labelsize=xlabels_ticks_size_axs[index_ax])
        set_sns_histplot_legend (ax, new_loc="lower left", prop = {"size" : 5}, title_fontsize=5)
        ax.axvline              (x=0.5, ls ='--', lw = 1., c = 'black')
    
    axs = clean_axs_of_violin_plots (axs)
    
    plt.savefig (dir_save + '/' + title, bbox_inches= 'tight')

#------------------------------------------------------------------------
# shape_continuous_value_in_block_for_df
def shape_continuous_value_in_block_for_df (df, continuous_name, block_name, rounds):

    rounds                          = np.array (rounds)
    values                          = np.array (df.loc [:, continuous_name])
    diff_continous_value_and_rounds = np.subtract.outer(values, rounds)
    index_in_rounds                 = np.argmin (abs(diff_continous_value_and_rounds), axis=1)
    block                           = rounds[index_in_rounds]
    unique, counts                  = np.unique(block, return_counts=True)
    print (continuous_name, dict(zip(unique, counts)))
    df[block_name]                  = block

    return df

#------------------------------------------------------------------------
