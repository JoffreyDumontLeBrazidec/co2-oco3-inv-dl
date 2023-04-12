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

from local_importeur                import *
from tools.tools_postproc           import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_list_colors
from folderPlot.presence_plots      import PresenceDataPostProc
from folderPlot.segmentation_plots  import SegmentationDataPostProc

#__________________________________________________________
# ManagerPostProc
class ManagerPostProc():

    #----------------------------------------------------
    # __initialiser__
    def __init__(self, list_plots, **kwargs):

        self.config                 = TreeConfigParser(comment_char='//')
        self.list_plots             = list_plots

    #------------------------------------------------------------------------
    # update_config
    def update_config(self, config_file):

        self.config.readfiles           (config_file)
        self.dir_save_current_model     = self.config.get("orga.save.directory") + "/" + self.config.get ("orga.save.folder")
        self.label                      = self.config.get('data.output.labelling')

    #------------------------------------------------------------------------
    # run_plots
    def run_plots(self):

        self.download_history_of_model()
        
        for plotIndicator in self.list_plots:
            print ('Plot:', plotIndicator)
            getattr(self, plotIndicator)()

    #------------------------------------------------------------------------
    # download_history_of_model
    def download_history_of_model(self):

        if (os.path.exists(self.dir_save_current_model + "/accuracy.bin")):
            self.accuracy       = np.array (np.fromfile(self.dir_save_current_model + "/" + "accuracy.bin"))
            self.val_accuracy   = np.array (np.fromfile(self.dir_save_current_model + "/" + "val_accuracy.bin"))
            
        self.loss           = np.array (np.fromfile(self.dir_save_current_model + "/" + "loss.bin"))
        self.val_loss       = np.array (np.fromfile(self.dir_save_current_model + "/" + "val_loss.bin"))
        self.N_epochs       = len(self.loss)

        if os.path.exists (self.dir_save_current_model + "/" + "test_metrics.bin"):
            test_metrics        = np.array (np.fromfile(self.dir_save_current_model + "/" + "test_metrics.bin"))
            self.test_loss      = test_metrics[0]
            if (len(test_metrics)>1):
                self.test_accuracy  = test_metrics[1]

    #------------------------------------------------------------------------
    # plot_loss
    def plot_loss(self):

        epochs_range = np.arange (0, self.N_epochs, 1)
        setMatplotlibParam() 
        list_colors = download_list_colors()
      
        fig             = plt.figure()
        sns.lineplot    (x=epochs_range, y=self.loss, color = list_colors[0], label = 'train')
        sns.lineplot    (x=epochs_range, y=self.val_loss, color = list_colors[1], label = 'validation')
        plt.scatter     (epochs_range[-1], self.test_loss, marker='o', s=100, color = "pink")       
        plt.ylabel      ('Loss')
        plt.xlabel      ('Epoch')
        plt.legend      (loc=2, fontsize = 'small')
        plt.savefig     (self.dir_save_current_model + '/' + 'loss' + '.png', bbox_inches= 'tight')
        plt.close       ()

    #------------------------------------------------------------------------
    # plot_accuracy
    def plot_accuracy(self):
        
        if self.label == "presence":
            
            epochs_range = np.arange (0, self.N_epochs, 1)
            setMatplotlibParam()
            list_colors = download_list_colors()

            fig             = plt.figure()
            sns.lineplot    (x=epochs_range, y=self.accuracy, color = list_colors[0], label = 'train')
            sns.lineplot    (x=epochs_range, y=self.val_accuracy, color = list_colors[1], label = 'validation')
            plt.scatter     (epochs_range[-1], self.test_accuracy, marker='o', s=100, color = "pink")
            plt.ylabel      ('Accuracy')
            plt.xlabel      ('Epoch')
            plt.legend      (loc=2, fontsize = 'small')
            plt.savefig     (self.dir_save_current_model + '/' + 'accuracy' + '.png', bbox_inches= 'tight')
            plt.close       ()

    #------------------------------------------------------------------------
    # deepDataExploration
    def deepDataExploration(self):

        if self.label == "presence":
            
            self.pres = PresenceDataPostProc(self.config)
            
            #self.pres.prepare_test_exploration_dataset(); 
            #self.pres.plot_wrong_predictions()
            #self.pres.plot_right_predictions()
            

            self.pres.download_df_test()
            #self.pres.make_violin_plots()
            #self.pres.plot_presence_confusion_matrix()
            self.pres.make_hist_plots()

        if self.label == "segmentation":
            
            self.seg = SegmentationDataPostProc(self.config)
            self.seg.prepare_test_exploration_dataset()
            self.seg.plot_predictions()
            
            self.seg.download_df_test()

#__________________________________________________________


