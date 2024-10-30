#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of data_plots
# TODO: 
#
#

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from local_importeur                    import *

from tools.tools_postproc               import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_color_map, clean_axs_of_violin_plots, set_sns_histplot_legend
from data.Data                          import Data
   
#------------------------------------------------------------------------
# import_data
def import_data(config, shuffleIndices):

    data                           = Data(config)
    data.prepareXCO2Data           (shuffleIndices = shuffleIndices)
    data.download_tt_dataset       ()
    return data

#------------------------------------------------------------------------
# plot_field
def plot_field(dir_save, array, tt_array, df_infos, index_image, title = ""):

    setMatplotlibParam()
    fig                 = plt.figure(figsize=(16, 9))
    ax                  = fig.add_axes([0.1, 0.1, 0.75, 0.75])
    plt.axes            (ax)
    cax                 = plt.axes([0.9,0.1,0.025,0.8])
    cax.set_title       (r'XCO2 in ppmv', size=25)

    c                   = ax.pcolor(array.transpose(1,0), cmap=download_color_map(), edgecolor = "face", zorder=0)
    c2                  = ax.pcolormesh(tt_array.transpose(1,0), cmap="viridis", shading='gouraud', alpha=0.35, zorder=1)
    cbar                = plt.colorbar(c2, cax, orientation='vertical')
    cbar.ax.tick_params (labelsize='25')

    ax.set_yticklabels  ([])
    ax.set_xticklabels  ([])
    ax.set_title        (title, fontsize=22, pad = 10)
    plt.savefig         (dir_save + '/' + 'test_image_' + str(index_image) + '.png')
    plt.close           ()

#------------------------------------------------------------------------

# plot_mix_fields
def plot_mix_fields(self, dir_save, array, tt_array, truth_binary_tt_array, pred_binary_tt_array, df_infos, index_image, title = ""):

    setMatplotlibParam()
    [ax1, ax2, ax3, ax4] = setFigure_2_2()
    axs = [ax1, ax2, ax3, ax4]

    c1                  = ax1.pcolor(array.transpose(1,0), cmap=download_color_map(), edgecolor = "face", zorder=0)
    c2                  = ax2.pcolor(tt_array.transpose(1,0), cmap="viridis", alpha=0.35, zorder=0)
    c3                  = ax3.pcolor(truth_binary_tt_array.transpose(1,0), cmap="viridis", alpha=0.35, zorder=0)       
    c4                  = ax4.pcolor(pred_binary_tt_array.transpose(1,0), cmap="viridis", alpha=0.35, zorder=0)              
        
    ax1.grid(False)  
    ax2.grid(False) 
    ax3.grid(False) 
    ax4.grid(False) 

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider             = make_axes_locatable(ax4)
    cax                 = divider.append_axes("right", size="5%", pad=0.05)
    cbar                = plt.colorbar(c4, cax, orientation='vertical')
    cbar.ax.tick_params (labelsize='7')

    #ax_title = fig.add_subplot(212, frameon = False)
    ax3.set_title           (title, fontsize=7, loc="right")
    plt.savefig         (dir_save + '/' + 'test_image_' + str(index_image) + '.png')
    
#------------------------------------------------------------------------