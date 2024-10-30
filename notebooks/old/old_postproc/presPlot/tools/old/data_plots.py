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
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib_functions import *
   
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
