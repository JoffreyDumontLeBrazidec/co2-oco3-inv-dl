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
from codestools_postproc    import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2
from test_manager           import TestPostProc
from data.Data              import Data

#__________________________________________________________
# ChainPostProc
class ChainPostProc():

    #----------------------------------------------------
    # __initialiser__
    def __init__(self, input_folder, first_folder = -9999, last_folder = 9999, **kwargs):

        # orga
        self.config         = TreeConfigParser(comment_char='//')
        self.input_folder   = input_folder

        self.general_dir_save_models    = input_folder + "/"
        self.df                         = pd.read_csv(self.general_dir_save_models + "/" + "table_combinations.csv")
        self.N_configs                  = self.df.shape[0]

        self.first_folder    = int (np.max ((0, first_folder)))
        self.last_folder     = int (np.min ((self.N_configs, last_folder)))
        self.number_folders  = self.last_folder - self.first_folder
    
        test_postproc = TestPostProc()        
        

    #------------------------------------------------------------------------
    # run_plots
    def run_plots(self, list_plots):

        if os.path.isfile (self.input_folder):

            self.download_history_of_model()
            self.plot_accuracy_and_loss()
            self.download_test_data(config)
            self.plot_confusion_matrix()
            self.plot_test_dataset_exploration(config)

        if os.path.isdir(self.input_folder):
        
            for plotIndicator in list_plots:
                print ('Plot:', plotIndicator)
                getattr(self, plotIndicator)()

    
    #------------------------------------------------------------------------
    # make_csv_history_table 
    def make_csv_history_table(self):
        
        self.list_histories_best_values = np.ones ((self.number_folders, 5)) * (-999)
        self.list_histories_last_values = np.ones ((self.number_folders, 4)) * (-999)

        for index_config in range (self.first_folder, self.last_folder):

            print ('index_config:', index_config)
            self.dir_save_current_model  = self.general_dir_save_models + "/test%s/"%index_config

            if (os.path.exists(self.dir_save_current_model + "/accuracy.bin")): # numpy array "proxy" appearing only when test completed
                
                self.download_history_of_model()
                self.fill_history_of_model(index_config)
            
        np.savetxt (self.general_dir_save_models + "/" + "best_values_history.txt", self.list_histories_best_values, fmt='%.2f',delimiter='\t')
        np.savetxt (self.general_dir_save_models + "/" + "last_values_history.txt", self.list_histories_last_values, fmt='%.2f',delimiter='\t')
    
        self.fill_and_save_complete_history()

    #------------------------------------------------------------------------
    # plot_accuracy_and_loss
    def plot_accuracy_and_loss(self):

        for index_config in range (self.first_folder, self.last_folder):

            print ('accuracy_loss:index_config:', index_config)
            self.dir_save_current_model  = self.general_dir_save_models + "/test%s/"%index_config

            if (os.path.exists(self.dir_save_current_model + "/accuracy.bin")): # numpy array "proxy" appearing only when test completed
                
                self.download_history_of_model()
                self.plot_run_accuracy_and_loss()

    #------------------------------------------------------------------------
    # download_history_of_model
    def download_history_of_model(self):

        self.accuracy       = np.array (np.fromfile(self.dir_save_current_model + "/" + "accuracy.bin"))
        self.loss           = np.array (np.fromfile(self.dir_save_current_model + "/" + "loss.bin"))
        self.val_accuracy   = np.array (np.fromfile(self.dir_save_current_model + "/" + "val_accuracy.bin"))
        self.val_loss       = np.array (np.fromfile(self.dir_save_current_model + "/" + "val_loss.bin"))
        self.N_epochs       = len(self.accuracy)

        if os.path.exists (self.dir_save_current_model + "/" + "test_metrics.bin"):
            test_metrics = np.array (np.fromfile(self.dir_save_current_model + "/" + "test_metrics.bin"))
            self.test_accuracy = test_metrics[1]
            self.test_loss = test_metrics[0]

    #------------------------------------------------------------------------
    # fill_history_of_model
    def fill_history_of_model(self, index_config):

        self.list_histories_best_values [index_config, 0]   = np.max (self.accuracy)
        self.list_histories_best_values [index_config, 1]   = np.min (self.loss)
        self.list_histories_best_values [index_config, 2]   = np.max (self.val_accuracy)
        self.list_histories_best_values [index_config, 3]   = np.min (self.val_loss)
        self.list_histories_best_values [index_config, 4]   = np.min (self.N_epochs)

        self.list_histories_last_values [index_config, 0]   = self.accuracy[-1]
        self.list_histories_last_values [index_config, 1]   = self.loss[-1]
        self.list_histories_last_values [index_config, 2]   = self.val_accuracy[-1]
        self.list_histories_last_values [index_config, 3]   = self.val_loss[-1]

    #------------------------------------------------------------------------
    # fill_and_save_complete_history
    def fill_and_save_complete_history(self):

        if "model.name" in self.df.columns:
            self.df = self.df.drop('model.name',axis=1)

        n_values_to_add                 = self.df.shape[0] - self.list_histories_best_values.shape[0] 
        self.df["accuracy.best"]        = np.pad(self.list_histories_best_values [:, 0], (0, n_values_to_add), 'constant', constant_values=(4, -999))
        self.df["loss.best"]            = np.pad(self.list_histories_best_values [:, 1], (0, n_values_to_add), 'constant', constant_values=(4, -999))
        self.df["val_accuracy.best"]    = np.pad(self.list_histories_best_values [:, 2], (0, n_values_to_add), 'constant', constant_values=(4, -999))
        self.df["val_loss.best"]        = np.pad(self.list_histories_best_values [:, 3], (0, n_values_to_add), 'constant', constant_values=(4, -999))
       
        for column_name in ["accuracy.best", "loss.best", "val_accuracy.best", "val_loss.best"]:
            self.df[column_name] = self.df[column_name].round(3)

        self.df["N_epochs"] = np.pad(self.list_histories_best_values [:, 4], (0, n_values_to_add), 'constant', constant_values=(4, -999)).astype("int")

        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]
        self.df.to_csv (self.general_dir_save_models + "/" +  "table_combinations.csv",  index=False)


#__________________________________________________________


