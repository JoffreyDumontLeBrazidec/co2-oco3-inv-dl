#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of presence_df_fill
# TODO: 
#
#

from local_importeur                    import *
from tools.tools_postproc               import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_color_map, clean_axs_of_violin_plots, set_sns_histplot_legend
from tools.tools_algebra                import find_nearest
from data.Data                          import Data
from tools.cm.pretty_confusion_matrix   import pp_matrix
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from config                             import config

# config
dir_save_current_model = config.get("orga.save.directory") + "/" + config.get ("orga.save.folder")
path_to_dataset        = config.get ('data.directory.main') + config.get ('data.directory.name')
config_dataset              = TreeConfigParser()
name_config_dataset         = config.get ("data.directory.name") + ".cfg"
config_dataset.readfiles    (path_to_dataset + "/" + name_config_dataset)
N_images               = config_dataset.get_int("N_images")
Ny                     = config_dataset.get_int("Ny")
Nx                     = config_dataset.get_int("Nx")
    
    #------------------------------------------------------------------------
    # import_data
    def import_data(self):

        self.shuffleIndices                 = np.array (np.fromfile(self.dir_save_current_model + "/" + "shuffle_indices.bin")).astype("int")
        self.data                           = Data(self.config)
        self.data.prepareXCO2Data           (shuffleIndices = self.shuffleIndices)
        self.data.download_tt_dataset       ()
        self.list_test_indices_considered   = list(self.shuffleIndices [self.data.N_trainingData+self.data.N_validationData:])
        self.x_test                         = self.data.x_test[0]
        self.y_test                         = self.data.y_test
        self.tt_test                        = self.data.tt_test

    #------------------------------------------------------------------------
    # make_test_predictions 
    def make_test_predictions(self):
        
        PDM_model                           = keras.models.load_model(self.dir_save_current_model + "/" + "PDM.h5")
        self.test_predictions               = PDM_model.predict (self.x_test)
        self.binary_test_predictions        = np.around (self.test_predictions)

    #------------------------------------------------------------------------
    # prepare_test_exploration_dataset
    def prepare_test_exploration_dataset(self):

        self.import_data()

        dir_infos_dataset               = self.config.get ("data.directory.main") + self.config.get ("data.directory.name")
        df                              = pd.read_pickle(dir_infos_dataset + "/" + "df_infos_dataset.plk")
        print ('self.df_test originel', df)
       
        self.df_test                    = df.loc [self.list_test_indices_considered]
        print ('self.df_test originel', self.df_test)
        self.make_test_predictions()
        self.df_test ['pred_pos_percentage'] = self.test_predictions
        test_prediction_success = abs(abs (np.squeeze(self.binary_test_predictions) - self.y_test) - 1).astype('bool')
        self.df_test ['pred_success'] = test_prediction_success
        
        tt_test_mean = self.tt_test.mean (axis=(1,2))
        self.df_test ['tt_test_mean'] = tt_test_mean

        self.tt_test[:,0,0] = -0.01
        
        tt_test_plume_mean = np.average(self.tt_test, axis=(1,2), weights=self.tt_test.astype(bool))
        self.df_test ['tt_test_plume_mean'] = tt_test_plume_mean

        Z = np.array([self.tt_test[i]-tt_test_plume_mean[i] for i in range (self.tt_test.shape[0])])
        tt_test_plume_var = np.average(Z**2, axis = (1,2), weights=self.tt_test.astype(bool))
        self.df_test ['tt_test_plume_var'] = tt_test_plume_var

        background_test_var = (self.x_test-self.tt_test).var (axis=(1,2))
        self.df_test ['ratio_var'] = tt_test_plume_var / background_test_var
        
        distance_to_truth = abs (np.squeeze(self.test_predictions) - self.y_test)
        self.df_test ['distance_to_truth'] = distance_to_truth

        self.df_test.to_csv (self.dir_save_current_model + "/" + "list_infos_predictions.csv")
   
    #------------------------------------------------------------------------
    # download_df_test
    def download_df_test(self):
 
        self.df_test = pd.read_csv (self.dir_save_current_model + "/" + "list_infos_predictions.csv", index_col=0)

    #------------------------------------------------------------------------
    # plot_presence_confusion_matrix
    def plot_presence_confusion_matrix(self):
        
        df = self.df_test
        
        dic_positivity = {"positive" : 1, "negative" : 0}
        df = df.replace({"positivity": dic_positivity})
        y_test = df.loc [:,"positivity"]
        
        binary_test_predictions = np.around(df.loc[:,"pred_pos_percentage"].to_numpy())

        from sklearn.metrics    import confusion_matrix
        cm                  = confusion_matrix(y_test, binary_test_predictions, labels = np.array([0,1]))
        df_cm               = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])  

        plt.rcdefaults      ()
        cmap                = "Oranges"
        pp_matrix           (df_cm, cmap=cmap)
        plt.savefig         (self.dir_save_current_model + "/" + "confusion_matrix_3" + ".png", bbox_inches = "tight")        
        plt.close           ()
        
    #------------------------------------------------------------------------
    # plot_wrong_predictions
    def plot_wrong_predictions(self):
 
        dir_plot_fields_wrongly_predicted   = self.dir_save_current_model + "/" + "images_wrong_predictions"
        pd_tools.createDirSmashingOldOne    (dir_plot_fields_wrongly_predicted)
        print ('df_test', self.df_test)
        index_in_df_test = np.arange (0, len(self.df_test))
        self.df_test ['index_in_df_test'] = index_in_df_test

        df_wrong_predictions                = self.df_test.loc[self.df_test['pred_success'] == False] 
        print ('df_wrong_predictions', df_wrong_predictions)       
        
        number_random_choices = 30
        import random
        for index_wrong_prediction in random.choices(list(df_wrong_predictions.index), k=number_random_choices):

            index_wrong_prediction_in_test = df_wrong_predictions.loc [index_wrong_prediction, 'index_in_df_test']
            current_x_test  = self.x_test [index_wrong_prediction_in_test,:,:,0]
            current_tt_test = self.tt_test [index_wrong_prediction_in_test,:,:,0]
            title_plot = "Truth:" + str(df_wrong_predictions.loc [index_wrong_prediction, "positivity"])            \
                        + "; dtt:" + "%.3f"%df_wrong_predictions.loc [index_wrong_prediction, "distance_to_truth"]   \
                        + "; cropping: " + str (df_wrong_predictions.loc [index_wrong_prediction, "cropping"])      \
                        + "; t: " + str (df_wrong_predictions.loc [index_wrong_prediction, "time"] % 24) + "h"
            self.plot_field (dir_plot_fields_wrongly_predicted, current_x_test, current_tt_test, df_wrong_predictions.loc[index_wrong_prediction], index_wrong_prediction, title = title_plot)

        df_wrong_predictions.to_csv (self.dir_save_current_model + "/" + "list_infos_wrong_predictions.csv")

    #------------------------------------------------------------------------
    # plot_right_predictions
    def plot_right_predictions(self):
 
        dir_plot_fields_rightly_predicted   = self.dir_save_current_model + "/" + "images_right_predictions"
        pd_tools.createDirSmashingOldOne    (dir_plot_fields_rightly_predicted)
        
        index_in_df_test = np.arange (0, len(self.df_test))
        self.df_test ['index_in_df_test'] = index_in_df_test

        df_right_predictions                = self.df_test.loc[self.df_test['pred_success'] == True] 
        
        number_random_choices = 30
        import random
        for index_right_prediction in random.choices(list(df_right_predictions.index), k=number_random_choices):

            index_right_prediction_in_test = df_right_predictions.loc [index_right_prediction, 'index_in_df_test']
            current_x_test  = self.x_test [index_right_prediction_in_test,:,:,0]
            current_tt_test = self.tt_test [index_right_prediction_in_test,:,:,0]
            title_plot = "Truth:" + str(df_right_predictions.loc [index_right_prediction, "positivity"])            \
                        + "; dtt:" + "%.3f"%df_right_predictions.loc [index_right_prediction, "distance_to_truth"]   \
                        + "; cropping: " + str (df_right_predictions.loc [index_right_prediction, "cropping"])      \
                        + "; t: " + str (df_right_predictions.loc [index_right_prediction, "time"] % 24) + "h"
            self.plot_field (dir_plot_fields_rightly_predicted, current_x_test, current_tt_test, df_right_predictions.loc[index_right_prediction], index_right_prediction, title = title_plot)

    #------------------------------------------------------------------------
    # plot_field
    def plot_field(self, dir_save, array, tt_array, df_infos, index_image, title = ""):

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
    # make_violin_plots
    def make_violin_plots(self):
 
        df = self.df_test 
        df = self.prepare_dataframe_with_hour_and_output (df)
        
        setMatplotlibParam()
        [ax1, ax2, ax3, ax4] = setFigure_2_2()
        axs = [ax1, ax2, ax3, ax4]
        
        variable_axs    = ["folder", "cropping", "hour_block", "positivity"]
        xlabel_axs      = ["Data origin", "Cropping binary", "Range beg. hour", "NN output"]
        text_axs        = ["(a)", "(b)", "(c)", "(d)"]
        ylabel_axs      = ["Distance to truth", "", "Distance to truth", ""]

        xlabels_ticks_size_axs    = [3,6,6,6]

        output          = "distance_to_truth"
        for index_ax, ax in zip (range(len(axs)), axs):
            
            sns.violinplot      (x=variable_axs[index_ax], y=output, data=df, ax=ax, inner='quartile', palette='Set2', saturation=0.8)
            ax.text             (0.05, 0.94, text_axs[index_ax], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xlabel       (xlabel_axs[index_ax])
            ax.set_ylabel       (ylabel_axs[index_ax])
            ax.tick_params      (axis='x', which='major', labelsize=xlabels_ticks_size_axs[index_ax])

        axs = clean_axs_of_violin_plots (axs)

        plt.savefig (self.dir_save_current_model + '/' + "/impact_pred_success_violing_plots.pdf", bbox_inches= 'tight')
        plt.close()
 
    #------------------------------------------------------------------------
    # make_hist_plots
    def make_hist_plots(self):
 
        # df full
        title = "impact_pred_success_hist_plots.pdf"

        df = self.df_test.copy()
        df = self.prepare_dataframe_with_hour_and_output (df)
        df.loc [df.distance_to_truth<0.05, "distance_to_truth"] = 0.05
        
        variable_axs    = ["folder", "cropping", "hour_block", "positivity"]
        xlabel_axs      = ["", "", "distance to truth", "distance to truth"]
        text_axs        = ["(a)", "(b)", "(c)", "(d)"]
        ylabel_axs      = ["Ind. normed density", "", "Ind. normed density", ""]
        output          = "distance_to_truth"
        xlabels_ticks_size_axs    = [6,6,6,6]

        self.make_histplots_with_var (df, output, variable_axs, xlabel_axs, text_axs, ylabel_axs, xlabels_ticks_size_axs, title)

        # df pos
        title = "impact_positive_pred_success_hist_plots.pdf"

        df_pos = self.df_test.copy()
        df_pos = df_pos.loc[df_pos['positivity'] == "positive"]  
        self.shape_continuous_value_in_block_for_df (df_pos, "tt_test_mean", "tt_mean_block", [0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.shape_continuous_value_in_block_for_df (df_pos, "tt_test_plume_var", "tt_plume_var_block", [0, 0.05, 0.1, 0.2])
        self.shape_continuous_value_in_block_for_df (df_pos, "ratio_var", "ratio_var_block", [0, 0.5, 1, 2, 3])
        self.shape_continuous_value_in_block_for_df (df_pos, "tt_test_plume_mean", "tt_plume_mean_block", [0,0.1, 0.2, 0.3, 0.4])

       
        df_pos.loc [df_pos.distance_to_truth<0.05, "distance_to_truth"] = 0.05
        df_pos.to_csv (self.dir_save_current_model + "/" + "list_infos_pos_predictions.csv")
        print (df_pos)
       
        variable_axs    = ["tt_mean_block", "tt_plume_var_block", "ratio_var_block", "tt_plume_mean_block"]
        xlabel_axs      = ["", "", "distance to truth", "distance to truth"]
        text_axs        = ["(a)", "(b)", "(c)", "(d)"]
        ylabel_axs      = ["Ind. normed density", "", "Ind. normed density", ""]
        output          = "distance_to_truth"
        xlabels_ticks_size_axs    = [6,6,6,6]
        
        self.make_histplots_with_var (df_pos, output, variable_axs, xlabel_axs, text_axs, ylabel_axs, xlabels_ticks_size_axs, title)

        # df pos non cropped
        title = "impact_noncropped_positive_pred_success_hist_plots.pdf"

        df_nc_pos = self.df_test.copy()
        df_nc_pos = df_nc_pos.loc[df_nc_pos['positivity'] == "positive"]  
        df_nc_pos = df_nc_pos.loc[df_nc_pos['cropping'] == False]
        self.shape_continuous_value_in_block_for_df (df_nc_pos, "tt_test_mean", "tt_mean_block", [0, 0.05, 0.1])
        self.shape_continuous_value_in_block_for_df (df_nc_pos, "tt_test_plume_var", "tt_plume_var_block", [0, 0.025, 0.05])
        self.shape_continuous_value_in_block_for_df (df_nc_pos, "ratio_var", "ratio_var_block", [0, 0.25, 0.5, 1])
        self.shape_continuous_value_in_block_for_df (df_nc_pos, "tt_test_plume_mean", "tt_plume_mean_block", [0,0.05, 0.1])

       
        df_nc_pos.loc [df_nc_pos.distance_to_truth<0.05, "distance_to_truth"] = 0.05
        df_nc_pos.to_csv (self.dir_save_current_model + "/" + "list_infos_pos_predictions.csv")
        print (df_nc_pos)
        
        variable_axs    = ["tt_mean_block", "tt_plume_var_block", "ratio_var_block", "tt_plume_mean_block"]
        xlabel_axs      = ["", "", "distance to truth", "distance to truth"]
        text_axs        = ["(a)", "(b)", "(c)", "(d)"]
        ylabel_axs      = ["Ind. normed density", "", "Ind. normed density", ""]
        output          = "distance_to_truth"
        xlabels_ticks_size_axs    = [6,6,6,6]
        
        self.make_histplots_with_var (df_nc_pos, output, variable_axs, xlabel_axs, text_axs, ylabel_axs, xlabels_ticks_size_axs, title)


    #------------------------------------------------------------------------   
    # make_histplots_with_var
    def make_histplots_with_var (self, df, output, variable_axs, xlabel_axs, text_axs, ylabel_axs, xlabels_ticks_size_axs, title):

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

        plt.savefig (self.dir_save_current_model + '/' + title, bbox_inches= 'tight')
        plt.close()
   
    #------------------------------------------------------------------------
    # prepare_dataframe_with_hour_and_output
    def prepare_dataframe_with_hour_and_output (self, df):

        hour = [None] * len(df.index.values)
        hour_block = [None] * len(df.index.values)       
        for index_test, norm_index_test in zip(df.index.values, range (len(df.index.values))):
            hour [norm_index_test] = df.at [index_test, "time"] % 24
            hour_block [norm_index_test] = int(np.trunc (hour [norm_index_test] / 4) * 4)
        df["hour"] = hour
        df["hour_block"] = hour_block
       
        self.shape_continuous_value_in_block_for_df (df, "pred_pos_percentage", "pred_pos_percentage_block", [0,0.2,0.4,0.6,0.8,1])
       
        return df

    #------------------------------------------------------------------------
    # shape_continuous_value_in_block_for_df
    def shape_continuous_value_in_block_for_df (self, df, continuous_name, block_name, rounds):

        rounds                          = np.array (rounds)
        values                          = np.array (df.loc [:, continuous_name])
        diff_continous_value_and_rounds = np.subtract.outer(values, rounds)
        index_in_rounds                 = np.argmin (abs(diff_continous_value_and_rounds), axis=1)
        block                           = rounds[index_in_rounds]
        unique, counts                  = np.unique(block, return_counts=True)
        print (continuous_name, dict(zip(unique, counts)))
        df[block_name]                  = block
        
        return df

#__________________________________________________________


