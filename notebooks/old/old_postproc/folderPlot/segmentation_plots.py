#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
## Implementation of segmentation_plots
# TODO: 
#
#

from local_importeur                    import *
from tools.tools_postproc               import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_color_map, clean_axs_of_violin_plots
from data.Data                          import Data
from tools.cm.pretty_confusion_matrix   import pp_matrix

#__________________________________________________________
# SegmentationDataPostProc
class SegmentationDataPostProc():

    #----------------------------------------------------
    # __initialiser__
    def __init__(self, config, **kwargs):

        self.config                 = config
        self.dir_save_current_model = self.config.get("orga.save.directory") + "/" + self.config.get ("orga.save.folder")
        

        self.path_to_dataset        = self.config.get ('data.directory.main') + self.config.get ('data.directory.name')
        config_dataset              = TreeConfigParser()
        name_config_dataset         = self.config.get ("data.directory.name") + ".cfg"
        config_dataset.readfiles    (self.path_to_dataset + "/" + name_config_dataset)
        self.N_images               = config_dataset.get_int("N_images")
        self.Ny                     = config_dataset.get_int("Ny")
        self.Nx                     = config_dataset.get_int("Nx")
    
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
        self.N_test                         = len(self.x_test)

    #------------------------------------------------------------------------
    # make_test_predictions 
    def make_test_predictions(self):

        PDM_model                           = keras.models.load_model(self.dir_save_current_model + "/" + "PDM.h5")
        self.y_pred                         = PDM_model.predict (self.x_test)

    #------------------------------------------------------------------------
    # prepare_test_exploration_dataset
    def prepare_test_exploration_dataset(self):
                
        self.import_data()
        self.make_test_predictions()

        self.test_loss = np.empty (self.N_test)
        bce = tf.keras.losses.BinaryCrossentropy()
        for i in range (self.N_test):
            self.test_loss[i] = bce(self.y_test[i], self.y_pred[i]).numpy()
        
        dir_infos_dataset               = self.config.get ("data.directory.main") + self.config.get ("data.directory.name")
        df                              = pd.read_pickle(dir_infos_dataset + "/" + "df_infos_dataset.plk")
        self.df_test                    = df.loc [self.list_test_indices_considered]

        self.df_test ['pred_loss'] = self.test_loss

        self.df_test.to_csv (self.dir_save_current_model + "/" + "list_infos_predictions.csv")
    
    #------------------------------------------------------------------------
    # download_df_test
    def download_df_test(self):

        self.df_test = pd.read_csv (self.dir_save_current_model + "/" + "list_infos_predictions.csv", index_col=0)

    #------------------------------------------------------------------------
    # plot_predictions
    def plot_predictions(self):

        dir_plot_fields   = self.dir_save_current_model + "/" + "images"
        pd_tools.createDirSmashingOldOne    (dir_plot_fields)
        
        index_in_df_test = np.arange (0, len(self.df_test))
        self.df_test ['index_in_df_test'] = index_in_df_test

        number_random_choices = 30
        import random
        for index_prediction in random.choices(list(self.df_test.index), k=number_random_choices):
            
            index_prediction_in_test = self.df_test.loc [index_prediction, 'index_in_df_test']

            current_x_test  = self.x_test [index_prediction_in_test,:,:,0]
            current_tt_test = self.tt_test [index_prediction_in_test,:,:,0]
            current_y_test  = self.y_test [index_prediction_in_test,:,:,0]
            current_y_pred  = self.y_pred [index_prediction_in_test,:,:,0]

            title_plot = "Truth:" + str(self.df_test.loc [index_prediction, "positivity"])            \
                        + "; bce:" + "%.3f"%self.df_test.loc [index_prediction, "pred_loss"]   \
                        + "; cropping: " + str (self.df_test.loc [index_prediction, "cropping"])      \
                        + "; t: " + str (self.df_test.loc [index_prediction, "time"] % 24) + "h"
            self.plot_mix_fields (dir_plot_fields, current_x_test, current_tt_test, current_y_test, current_y_pred, self.df_test.loc[index_prediction], index_prediction, title = title_plot)

    #------------------------------------------------------------------------
    # plot_mix_fields
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
        plt.close           ()

#__________________________________________________________



