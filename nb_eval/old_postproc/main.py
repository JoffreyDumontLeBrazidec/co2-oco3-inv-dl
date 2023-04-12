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

from local_importeur        import *
from folderPlot.manager     import ManagerPostProc

if __name__ == "__main__":

    
    list_plots = ['plot_loss', 'plot_accuracy', 'deepDataExploration']

    in_command = bash.config_file_names_from_command()

    if (len(in_command) == 0):
        config_file             = "../model_weights/chaintest2/test0/config.cfg"

    elif (len(in_command) == 1):
        config_file = in_command

    pp                = ManagerPostProc(list_plots = list_plots)
    pp.update_config  (config_file)
    pp.run_plots      ()
