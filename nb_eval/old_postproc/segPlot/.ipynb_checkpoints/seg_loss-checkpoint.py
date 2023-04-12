#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of seg_loss
# TODO: 
#
#

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from pres_config                import *
from local_importeur            import *

from tools.tools_postproc           import setMatplotlibParam, setFigure_2, setFigure_2_1, setFigure_2_2, setFigure_2_2_1, setFigure_2_2_2, download_list_colors

# download loss and accuracy
loss           = np.array (np.fromfile(dir_save_current_model + "/" + "loss.bin"))
val_loss       = np.array (np.fromfile(dir_save_current_model + "/" + "val_loss.bin"))
N_epochs       = len(loss)

test_metrics    = np.array (np.fromfile(dir_save_current_model + "/" + "test_metrics.bin"))
test_loss       = test_metrics[0]

# prepare plots
epochs_range = np.arange (0, N_epochs, 1)
setMatplotlibParam() 
list_colors = download_list_colors()

# plot loss
fig             = plt.figure()
sns.lineplot    (x=epochs_range, y=loss, color = list_colors[0], label = 'train')
sns.lineplot    (x=epochs_range, y=val_loss, color = list_colors[1], label = 'validation')
plt.scatter     (epochs_range[-1], test_loss, marker='o', s=100, color = "pink")       
plt.ylabel      ('Loss')
plt.xlabel      ('Epoch')
plt.legend      (loc=2, fontsize = 'small')
plt.savefig     (dir_save_current_model + '/' + 'loss' + '.png', bbox_inches= 'tight')

#__________________________________________________________


