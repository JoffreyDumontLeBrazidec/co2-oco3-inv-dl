#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of tools_postproc
# TODO: 
#
#

from importeur import *
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

#------------------------------------------------------------------------
# setMatplotlibParam
def setMatplotlibParam():

    sns.set_context     ('paper')
    sns.set_style       ('whitegrid')
        
    plt.rc('lines',         linewidth           = 0.66)
    plt.rc('lines',         markeredgewidth     = 0.5)
    plt.rc('lines',         markersize          = 4)
    plt.rc('figure',        dpi                 = 300)
    plt.rc('font',          family              = 'serif')
    plt.rc('savefig',       format              = 'pdf')
    plt.rc('savefig',       facecolor           = 'white')
    plt.rc('axes',          linewidth           = 0.7)
    plt.rc('axes',          edgecolor           = 'k')
    plt.rc('axes',          facecolor           = [0.96, 0.96, 0.96])
    plt.rc('axes',          labelsize           = 'x-small')
    plt.rc('axes',          titlesize           = 'x-small')
    plt.rc('legend',        fontsize            = 'x-small')
    plt.rc('legend',        frameon             = True)
    plt.rc('legend',        framealpha          = 1)
    plt.rc('legend',        handlelength        = 3)
    plt.rc('legend',        numpoints           = 3)
    plt.rc('legend',        markerscale         = 1)
    plt.rc('xtick',         labelsize           = 'x-small')
    plt.rc('ytick',         labelsize           = 'x-small')
    plt.rc('xtick.major',   pad                 = 0)
    plt.rc('ytick.major',   pad                 = 0)

#------------------------------------------------------------------------
# download_list_colors
def download_list_colors ():

    list_colors    = ['firebrick', 'midnightblue', 'firebrick', 'blue', 'midnightblue','yellow', 'greenyellow', 'green']
    return list_colors

#------------------------------------------------------------------------
# download_color_map
def download_color_map ():

    FuncCmap = scipy.interpolate.interp1d(np.asarray([0.,0.2, 0.60, 0.8, 1]), np.asarray([[1, 1, 1 ,1], [149./255., 208./255., 252./255., 1.], [174./255., 113./255., 129./255., 1],\
                            [97./255., 0., 35./255., 1], [1., 2./255., 141./255., 1.]]).T)
    color_map = ListedColormap(FuncCmap(np.linspace(0.,1., 1024)).T)
    return color_map

#------------------------------------------------------------------------
# setFigure_2
def setFigure_2 (pad_w_int=0.35):

    plt.close()

    wratio      = 0.35
    hratio      = 0.75
    pad_w_ext   = 0.3
    pad_h_ext   = 0.05

    # linewidth
    linewidth   = 5.80910486111 # [inch]

    # (w, h) for the ax
    ax_w        = wratio * linewidth
    ax_h        = hratio * ax_w

    # (w, h) for the figure
    fig_w       = pad_w_ext + ax_w + pad_w_int + ax_w + pad_w_ext
    fig_h       = pad_h_ext + ax_h + pad_h_ext

    # (x, y) for the ax 
    ax1_x       = pad_w_ext / fig_w
    ax2_x       = ( pad_w_ext + ax_w + pad_w_int ) / fig_w
    ax2_y       = pad_h_ext / fig_h
    ax1_y       = pad_h_ext / fig_h
    ax_dx       = ax_w / fig_w
    ax_dy       = ax_h / fig_h

    # create figure and ax
    figure      = plt.figure(figsize=(fig_w, fig_h))
    ax1         = figure.add_axes([ax1_x, ax1_y, ax_dx, ax_dy])
    ax2         = figure.add_axes([ax2_x, ax1_y, ax_dx, ax_dy])

    axs         = [ax1, ax2]

    return axs 

#------------------------------------------------------------------------
# setFigure_2_1
def setFigure_2_1 (pad_w_int=0.35, pad_h_int=0.33):

    plt.close()

    wratio      = 0.35
    hratio      = 0.75
    pad_w_ext   = 0.3
    pad_h_ext   = 0.05

    # linewidth
    linewidth   = 5.80910486111 # [inch]

    # (w, h) for the ax
    ax_w        = wratio * linewidth
    ax_h        = hratio * ax_w

    # (w, h) for the figure
    fig_w       = pad_w_ext + ax_w + pad_w_int + ax_w + pad_w_ext
    fig_h       = pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_ext

    # (x, y) for the ax 
    ax1_x       = pad_w_ext / fig_w
    ax2_x       = ( pad_w_ext + ax_w + pad_w_int ) / fig_w
    ax2_y       = pad_h_ext / fig_h
    ax1_y       = ( pad_h_ext  + ax_h + pad_h_int ) / fig_h
    ax_dx       = ax_w / fig_w
    ax_dy       = ax_h / fig_h

    # create figure and ax
    figure      = plt.figure(figsize=(fig_w, fig_h))
    ax1         = figure.add_axes([ax1_x, ax1_y, ax_dx, ax_dy])
    ax2         = figure.add_axes([ax2_x, ax1_y, ax_dx, ax_dy])
    ax3         = figure.add_axes([ax1_x, ax2_y, ax_dx, ax_dy])

    axs         = [ax1, ax2, ax3]

    return axs 

#------------------------------------------------------------------------
# setFigure_2_2
def setFigure_2_2 (pad_w_int=0.35, pad_h_int=0.33):

    plt.close()

    wratio      = 0.35
    hratio      = 0.75
    pad_w_ext   = 0.3
    pad_h_ext   = 0.05

    # linewidth
    linewidth   = 5.80910486111 # [inch]

    # (w, h) for the ax
    ax_w        = wratio * linewidth
    ax_h        = hratio * ax_w

    # (w, h) for the figure
    fig_w       = pad_w_ext + ax_w + pad_w_int + ax_w + pad_w_ext
    fig_h       = pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_ext

    # (x, y) for the ax 
    ax1_x       = pad_w_ext / fig_w
    ax2_x       = ( pad_w_ext + ax_w + pad_w_int ) / fig_w
    ax2_y       = pad_h_ext / fig_h
    ax1_y       = ( pad_h_ext  + ax_h + pad_h_int ) / fig_h
    ax_dx       = ax_w / fig_w
    ax_dy       = ax_h / fig_h

    # create figure and ax
    figure      = plt.figure(figsize=(fig_w, fig_h))
    ax1         = figure.add_axes([ax1_x, ax1_y, ax_dx, ax_dy])
    ax2         = figure.add_axes([ax2_x, ax1_y, ax_dx, ax_dy])
    ax3         = figure.add_axes([ax1_x, ax2_y, ax_dx, ax_dy])
    ax4         = figure.add_axes([ax2_x, ax2_y, ax_dx, ax_dy])

    axs         = [ax1, ax2, ax3, ax4]

    return axs 

#------------------------------------------------------------------------
# setFigure_2_2_1
def setFigure_2_2_1 (pad_w_int=0.35, pad_h_int=0.33):

    plt.close()

    wratio      = 0.35
    hratio      = 0.75
    pad_w_ext   = 0.3
    pad_h_ext   = 0.05

    # linewidth
    linewidth   = 5.80910486111 # [inch]

    # (w, h) for the ax
    ax_w        = wratio * linewidth
    ax_h        = hratio * ax_w

    # (w, h) for the figure
    fig_w       = pad_w_ext + ax_w + pad_w_int + ax_w + pad_w_ext
    fig_h       = pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_int + ax_h + pad_h_ext

    # (x, y) for the ax 
    ax1_x       = pad_w_ext / fig_w
    ax2_x       = ( pad_w_ext + ax_w + pad_w_int ) / fig_w
    ax3_y       = pad_h_ext / fig_h
    ax2_y       = ( pad_h_ext + ax_h + pad_h_int ) / fig_h
    ax1_y       = ( pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_int ) / fig_h
    ax_dx       = ax_w / fig_w
    ax_dy       = ax_h / fig_h

    # create figure and ax
    figure      = plt.figure(figsize=(fig_w, fig_h))
    ax1         = figure.add_axes([ax1_x, ax1_y, ax_dx, ax_dy])
    ax2         = figure.add_axes([ax2_x, ax1_y, ax_dx, ax_dy])
    ax3         = figure.add_axes([ax1_x, ax2_y, ax_dx, ax_dy])
    ax4         = figure.add_axes([ax2_x, ax2_y, ax_dx, ax_dy])
    ax5         = figure.add_axes([ax1_x, ax3_y, ax_dx, ax_dy])

    axs         = [ax1, ax2, ax3, ax4, ax5]

    return axs 

#------------------------------------------------------------------------
# setFigure_2_2_2
def setFigure_2_2_2 (pad_w_int=0.35, pad_h_int=0.33):

    plt.close()

    wratio      = 0.35
    hratio      = 0.75
    pad_w_ext   = 0.3
    pad_h_ext   = 0.01

    # linewidth
    linewidth   = 5.80910486111 # [inch]

    # (w, h) for the ax
    ax_w        = wratio * linewidth
    ax_h        = hratio * ax_w

    # (w, h) for the figure
    fig_w       = pad_w_ext + ax_w + pad_w_int + ax_w + pad_w_ext
    fig_h       = pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_int + ax_h + pad_h_ext

    # (x, y) for the ax 
    ax1_x       = pad_w_ext / fig_w
    ax2_x       = ( pad_w_ext + ax_w + pad_w_int ) / fig_w
    ax3_y       = pad_h_ext / fig_h
    ax2_y       = ( pad_h_ext + ax_h + pad_h_int ) / fig_h
    ax1_y       = ( pad_h_ext + ax_h + pad_h_int + ax_h + pad_h_int ) / fig_h
    ax_dx       = ax_w / fig_w
    ax_dy       = ax_h / fig_h

    # create figure and ax
    figure      = plt.figure(figsize=(fig_w, fig_h))
    ax1         = figure.add_axes([ax1_x, ax1_y, ax_dx, ax_dy])
    ax2         = figure.add_axes([ax2_x, ax1_y, ax_dx, ax_dy])
    ax3         = figure.add_axes([ax1_x, ax2_y, ax_dx, ax_dy])
    ax4         = figure.add_axes([ax2_x, ax2_y, ax_dx, ax_dy])
    ax5         = figure.add_axes([ax1_x, ax3_y, ax_dx, ax_dy])
    ax6         = figure.add_axes([ax2_x, ax3_y, ax_dx, ax_dy])

    axs         = [ax1, ax2, ax3, ax4, ax5, ax6]

    return axs 

#------------------------------------------------------------------------
# clean_axs_of_violin_plots
def clean_axs_of_violin_plots (axs):

    for ax in axs:
        for index_line, line in zip(range(len(ax.lines)), ax.lines):
            if index_line%3==0:
                line.set_linestyle('--')
                line.set_color('black')
                line.set_alpha(0.9)

            if (index_line+1)%3==0:
                line.set_linestyle('--')
                line.set_color('black')
                line.set_alpha(0.9)

            if (index_line+2)%3==0:
                line.set_linestyle('-')
                line.set_color('black')
                line.set_alpha(1)
                line.set_linewidth (1)
    
    return axs

#------------------------------------------------------------------------
# set_sns_histplot_legend
def set_sns_histplot_legend(ax, new_loc, **kws):
    old_legend  = ax.legend_
    handles     = old_legend.legendHandles
    labels      = [t.get_text() for t in old_legend.get_texts()]
    title       = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)

#__________________________________________________________


