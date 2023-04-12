# -------------------------------------------------
# dev/plumeDetection/postprocessing/main
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of folder_config
# TODO:
#
#

from treeconfigparser import TreeConfigParser
import os
import xarray as xr

base_dir = "/cerea_raid/users/dumontj/dev/coco2/dl/res"
config_file = "chaintest_2_20220420-201904/efficient_random/config.cfg"

# config
config_file = os.path.join(base_dir, config_file)
config = TreeConfigParser(comment_char="//")
config.readfiles(config_file)
label = config.get("data.output.labelling")

name_model = config.get("model.name")
dir_save_current_model = os.path.join(base_dir, config.get("orga.save.folder"))
path_to_dataset = os.path.join(
    config.get("data.directory.main"), config.get("data.directory.name"), "dataset.nc"
)

ds = xr.open_dataset(path_to_dataset)
N_images = ds.attrs["N_images"]
Ny = ds.y.shape[0]
Nx = ds.x.shape[0]
