# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_loss_acc
# TODO:
#
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("/cerea_raid/users/dumontj/dev/coco2/dl/postprocessing/")
from matplotlib_functions import *
from pres_config import *

# download loss and accuracy
accuracy = np.array(np.fromfile(dir_save_current_model + "/" + "accuracy.bin"))
val_accuracy = np.array(np.fromfile(dir_save_current_model + "/" + "val_accuracy.bin"))
loss = np.array(np.fromfile(dir_save_current_model + "/" + "loss.bin"))
val_loss = np.array(np.fromfile(dir_save_current_model + "/" + "val_loss.bin"))
N_epochs = len(loss)

test_metrics = np.array(np.fromfile(dir_save_current_model + "/" + "test_metrics.bin"))
test_loss = test_metrics[0]
test_accuracy = test_metrics[1]

# prepare plots
epochs_range = np.arange(0, N_epochs, 1)
setMatplotlibParam()
list_colors = download_list_colors()

# plot loss
fig = plt.figure()
sns.lineplot(x=epochs_range, y=loss, color=list_colors[0], label="train")
sns.lineplot(x=epochs_range, y=val_loss, color=list_colors[1], label="validation")
plt.scatter(epochs_range[-1], test_loss, marker="o", s=100, color="pink")
plt.ylabel("Loss", fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(loc=4, fontsize=12)
plt.savefig(dir_save_current_model + "/" + "loss" + ".png", bbox_inches="tight")
plt.close()

# plot accuracy
fig = plt.figure()
sns.lineplot(x=epochs_range, y=accuracy, color=list_colors[0], label="train")
sns.lineplot(x=epochs_range, y=val_accuracy, color=list_colors[1], label="validation")
plt.scatter(epochs_range[-1], test_accuracy, marker="o", s=100, color="pink")
plt.ylabel("Accuracy",fontsize=12)
plt.xlabel("Epoch", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.legend(loc=4, fontsize=12)
plt.savefig(dir_save_current_model + "/" + "accuracy" + ".png", bbox_inches="tight")
plt.close()

# __________________________________________________________
