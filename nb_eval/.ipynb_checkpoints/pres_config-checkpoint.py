#-------------------------------------------------
# dev/plumeDetection/postprocessing/main
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of folder_config
# TODO: 
#
#

from local_importeur        import *

base_dir = "/cerea_raid/users/dumontj/dev/plumeDetection/models/model_weights/"
config_file = "chaintest2/test3/config.cfg"

# config
config_file         = base_dir + "/" + config_file
config              = TreeConfigParser(comment_char='//')
config.readfiles    (config_file)
label               = config.get('data.output.labelling')

dir_save_current_model      = config.get("orga.save.directory") + "/" + config.get ("orga.save.folder")
path_to_dataset             = config.get ('data.directory.main') + config.get ('data.directory.name')
config_dataset              = TreeConfigParser()
name_config_dataset         = config.get ("data.directory.name") + ".cfg"
config_dataset.readfiles    (path_to_dataset + "/" + name_config_dataset)
N_images                    = config_dataset.get_int("N_images")
Ny                          = config_dataset.get_int("Ny")
Nx                          = config_dataset.get_int("Nx")





