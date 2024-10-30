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

from local_importeur        import *
from codes.tools_postproc   import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2


#------------------------------------------------------------------------
# make_violin_plot
def make_violin_plot (df, directory_plots):

    df = prepare_dataframe_with_inputs(df)
    
    setMatplotlibParam()
    [ax1, ax2, ax3] = setFigure_2_1()
    axs = [ax2, ax2, ax3]

    variable_axs    = ["input.time", "wind_as_input", "dynamic_as_input"]
    xlabel_axs      = ["Input time", "Input wind information", "Input dynamic information"]
    text_axs        = ["a", "b", "c"]
    ylabel_axs      = ["Validation dataset accuracy - best", "", "Validation dataset accuracy - best"]

    output          = "val_accuracy.best"

    for index_ax, ax in zip (len(axs), axs):
       
        sns.violinplot (x=variable_axs[index_ax], y=output, data=df, axs=ax, inner='quartile', palette='Set2', saturation=0.8)
        ax.text (0.05, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)    
        ax.set_xlabel(xlabel_axs[index_ax])
        ax.set_ylabel(ylabel_axs[index_ax])

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
    """
    leg = ax3.legend(loc=4, prop={'size': 4.5})
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    """

    plt.savefig (directory_plots + "/" + "/impact_on_val_accuracy_violing_plots_df_P1_N1.pdf", bbox_inches= 'tight')
    plt.close()

#------------------------------------------------------------------------
# prepare_dataframe_with_inputs
def prepare_dataframe_with_inputs (df):
 
    clean_df = df.drop (self.df[self.df["val_accuracy.best"] < -1].index)
    
    df_P1_N1 = clean_df.drop (clean_df[clean_df["dataset"] != "CO2_P1ALL_N1BB_tt06"].index)
    
    # wind_as_input
    wind_as_input = [None] * len(df_P1_N1.index.values)
    for index_test, norm_index_test in zip(df_P1_N1.index.values, range (len(df_P1_N1.index.values))):
        wind_as_input [norm_index_test] = df_P1_N1.at[index_test, "winds.format"]
        if wind_as_input [norm_index_test] == "fields":
            wind_as_input [norm_index_test] = "field" + "[" + str(df_P1_N1.at[index_test, "N_wind_fields"]) + "ch]"
    df_P1_N1["wind_as_input"] = wind_as_input 
    
    # dynamics_as_input
    dynamic_as_input = [None] * len(df_P1_N1.index.values)
    for index_test, norm_index_test in zip(df_P1_N1.index.values, range (len(df_P1_N1.index.values))):
        dynamic_as_input [norm_index_test] = df_P1_N1.at[index_test, "dynamic.format"]
        if dynamic_as_input [norm_index_test] == "fields":
            dynamic_as_input [norm_index_test] = "field" + "[" + str(df_P1_N1.at[index_test, "N_dynamic_fields"]) + "ch]"
    df_P1_N1["dynamic_as_input"] = dynamic_as_input 

    return df_P1_N1

#__________________________________________________________


