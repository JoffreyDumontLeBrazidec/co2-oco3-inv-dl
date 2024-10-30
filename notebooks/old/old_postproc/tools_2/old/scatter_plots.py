#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of postproc
# TODO: 
#
#

from local_importeur    import *
from tools_postproc     import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2


#------------------------------------------------------------------------
# make_violin_plot
def make_violin_plot (df, directory_plots):
   
    # set figure
    setMatplotlibParam()
    [ax1, ax2, ax3] = setFigure_2_1()
    axs = [ax2, ax2, ax3]

    clean_df = df.drop (self.df[self.df["val_accuracy.best"] < -1].index)

    df_P1_N1 = clean_df.drop (clean_df[clean_df["dataset"] != "CO2_P1ALL_N1BB_tt06"].index)
    # wind_as_input
    wind_as_input = [None] * len(df_P1_N1.index.values)
    for index_test, norm_index_test in zip(df_P1_N1.index.values, range (len(df_P1_N1.index.values))):
        print ('index_test', index_test)
        wind_as_input [norm_index_test] = df_P1_N1.at[index_test, "winds.format"]
        if wind_as_input [norm_index_test] == "fields":
            wind_as_input [norm_index_test] = "field" + "[" + str(df_P1_N1.at[index_test, "N_wind_fields"]) + "ch]"
    df_P1_N1["wind_as_input"] = wind_as_input 
    
    # dynamics_as_input
    dynamic_as_input = [None] * len(df_P1_N1.index.values)
    for index_test, norm_index_test in zip(df_P1_N1.index.values, range (len(df_P1_N1.index.values))):
        print ('index_test', index_test)
        dynamic_as_input [norm_index_test] = df_P1_N1.at[index_test, "dynamic.format"]
        if dynamic_as_input [norm_index_test] == "fields":
            dynamic_as_input [norm_index_test] = "field" + "[" + str(df_P1_N1.at[index_test, "N_dynamic_fields"]) + "ch]"
    df_P1_N1["dynamic_as_input"] = dynamic_as_input 

    # ax1
    output_ax1      = "val_accuracy.best"
    variable_ax1    = "input.time"
    sns.violinplot (x=variable_ax1, y=output_ax1, data=df_P1_N1, ax=ax1, inner = "quartile", palette="Set2", dodge=False, saturation = 0.8)
    
    ax1.text(0.05, 0.94, '(a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)    
    ax1.set_xlabel('Input Time')
    ax1.set_ylabel('Validation dataset accuracy - best')
    
    # ax2
    output_ax2      = "val_accuracy.best"
    variable_ax2    = "wind_as_input"
    sns.violinplot (x=variable_ax2, y=output_ax2, data=df_P1_N1, ax=ax2, inner = "quartile", palette="Set2", saturation = 0.8)
    
    ax2.text(0.05, 0.94, '(b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)    
    ax2.set_xlabel('Input Wind information')
    ax2.set_ylabel('')
    leg = ax2.legend(loc=3, prop={'size': 4.5})
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    # ax3
    output_ax3      = "val_accuracy.best"
    variable_ax3    = "dynamic_as_input"
    sns.violinplot (x=variable_ax3, y=output_ax3, data=df_P1_N1, ax=ax3, inner = "quartile", palette="Set2", saturation = 0.8)
    
    ax3.text(0.05, 0.94, '(c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)    
    ax3.set_xlabel('Input Dynamic information')
    ax3.set_ylabel('Validation dataset accuracy - best')
    leg = ax3.legend(loc=4, prop={'size': 4.5})
    for lh in leg.legendHandles:
        lh.set_alpha(1)


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

    plt.savefig (directory_plots + "/" + "/impact_on_val_accuracy_violing_plots_df_P1_N1.pdf", bbox_inches= 'tight')
    plt.close()

#__________________________________________________________


