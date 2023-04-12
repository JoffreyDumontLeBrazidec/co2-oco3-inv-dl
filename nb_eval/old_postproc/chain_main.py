#-------------------------------------------------
# dev/plumeDetection/postprocessing/main
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of main
# TODO: 
#
#

from local_importeur    import *
from basic_postproc     import postProcModelsResults
from scatter_plots      import make_violin_plot

if __name__ == "__main__":

    in_command  = bash.config_file_names_from_command()

    if (len(in_command) == 0):
        folder_or_configFile    = "../model_weights/chaintest1/"
        N_folders               = len (glob.glob(folder_or_configFile+"/test*/"))
        first_folder            = 0
        last_folder             = N_folders-1
        print                   ("postproc:", folder_or_configFile)

    elif (len(in_command) == 1):
        folder_or_configFile    = in_command[0] 
        N_folders               = len (glob.glob(folder_or_configFile+"/test*/"))
        first_folder            = 0
        last_folder             = N_folders-1
        print                   ("postproc:", folder_or_configFile)
    
    elif (len(in_command) == 3):
        folder_or_configFile    = in_command[0]    
        first_folder            = int (in_command[1])   
        last_folder             = int (in_command[2])
        print                   ("postproc:", folder_or_configFile)

    else:
        print ("command line problem")
        exit()


    #list_folders            = [directory_all_folders + '/' + 'test' + str(index_folder) for index_folder in range (N_folders)]
    postprocessor   = postProcModelsResults   (folder_or_configFile, first_folder = first_folder, last_folder = last_folder)

    #list_plots          = ['make_csv_history_table', 'plot_accuracy_and_loss']
    #postprocessor.run_plots (list_plots)
    #make_violin_plot (postprocessor.df, postprocessor.general_dir_save_models)
    postprocessor.download_test_data()
    postprocessor.plot_test_dataset_exploration()

