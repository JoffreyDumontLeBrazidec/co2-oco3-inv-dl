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

from importeur import *
from tensorflow import keras
from tools_postproc import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/')
from data.Data import Data
from codes.cm.pretty_confusion_matrix import pp_matrix

#__________________________________________________________
# PresenceDataPostProc
class PresenceDataPostProc():

    #----------------------------------------------------
    # __initialiser__
    def __init__(self, config, **kwargs):

        self.config                 = config
        self.dir_save_current_model = self.config.get("orga.save.directory") + "/" + self.config.get ("orga.save.folder")
   
    #------------------------------------------------------------------------
    # import_data
    def import_data(self):

        self.shuffleIndices         = np.array (np.fromfile(self.dir_save_current_model + "/" + "shuffle_indices.bin")).astype("int")
        self.data                   = Data(self.config)
        self.data.prepareXCO2Data   (shuffleIndices = self.shuffleIndices)

    #------------------------------------------------------------------------
    # make_test_predictions 
    def make_test_predictions(self):
        
        self.list_test_indices_considered       = list(self.shuffleIndices [self.data.N_trainingData+self.data.N_validationData:])
        self.x_test                         = self.data.x_test[0]
        self.y_test                         = self.data.y_test
        PDM_model                               = keras.models.load_model(self.dir_save_current_model + "/" + "PDM.h5")
        self.test_predictions               = PDM_model.predict (self.x_test)
        self.binary_test_predictions        = np.around (self.test_predictions)

    #------------------------------------------------------------------------
    # prepare_test_exploration_dataset
    def prepare_test_exploration_dataset(self):
 
        dir_infos_dataset               = self.config.get ("data.directory.main") + self.config.get ("data.directory.name")
        df                              = pd.read_pickle(dir_infos_dataset + "/" + "df_infos_dataset.plk")
        df_test                         = df.loc [self.list_test_indices_considered]

        df_test ['pred_pos_percentage'] = self.test_predictions
        test_prediction_success = abs(abs (np.squeeze(self.binary_test_predictions) - self.y_test) - 1).astype('bool')
        df_test ['pred_success'] = test_prediction_success
        
        df_test.to_csv (self.dir_save_current_model + "/" + "list_infos_predictions.csv")
    
    #------------------------------------------------------------------------
    # plot_presence_confusion_matrix
    def plot_presence_confusion_matrix(self):
        
        confusion_matrix                = sklearn.metrics.confusion_matrix(self.binary_test_predictions, self.y_test, labels = np.array([0,1]))
        df_cm                           = pd.DataFrame(confusion_matrix, index=[0, 1], columns=[0, 1])  

        plt.rcdefaults  ()
        cmap            = "Oranges"
        pp_matrix       (df_cm, cmap=cmap)
        plt.savefig     (self.dir_save_current_model + "/" + "confusion_matrix_3" + ".png", bbox_inches = "tight")        
        plt.close       ()
        
        """
        fig             = plt.figure()
        sns.set         (font_scale=1.4)
        sns.heatmap     (df_cm, annot=True, annot_kws={"size": 16})
        plt.ylabel      ('Actual label')
        plt.xlabel      ('Predicted label')
        plt.savefig     (self.dir_save_current_model + "/" + "confusion_matrix" + ".png", bbox_inches = "tight") 
        plt.close()

        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels = np.array([0,1]))
        disp.plot()
        plt.savefig (self.dir_save_current_model + "/" + "confusion_matrix_2" + ".png", bbox_inches = "tight") 
        plt.close()
        """
        # plotter ceux qui ont été mal prédits dans un folder dans le self.dir_save_current_model

    #------------------------------------------------------------------------
    # plot_all_false_predictions
    #def plot_all_false_predictions(self):
 
        # creer un dossier "images_false_predictions" dans le dossier model_weights/test(?)

        # reperer false predictions à partir de df_test

        # reperer le XCO2 field correspondant dans multipleInputs

        # prendre l'image et la plot dans le dossier model_weights/test(?)

    #------------------------------------------------------------------------
    # make_violin_plots
    #def make_violin_plots(self):
 
        # creer un dossier "images_false_predictions" dans le dossier model_weights/test(?)

        # violin plots of predictions success against the diverse variables (cropping or not, time night or day à retrouver, folder origin) 


#__________________________________________________________


