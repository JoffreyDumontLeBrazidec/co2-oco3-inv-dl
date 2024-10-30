#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of pres_loss_acc
# TODO: 
#
#

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.cm as cm
sys.path.append('../')
from chain_config import *
from matplotlib_functions import * 
import scipy.signal

# IL SERAIT PUISSANT DE POUVOIR SELECTIONNER EN FONCTION DU DATAFRAME.CSV LES DECRIVANT 
# LES DIFFERENTES CONFIGURATIONS AFIN DE LES PLOT

list_exp = [exp for exp in glob.glob(dir_all_models+"/*") if os.path.exists(os.path.join(exp, "accuracy.bin"))]

# selection
selected_exp = ["efficient_random", "res_random", "res_imagenet"]
list_exp = [exp for exp in list_exp if os.path.basename(os.path.normpath(exp)) in selected_exp]

N_exps = len(list_exp)
acc = [-999]*N_exps
val_acc = [-999]*N_exps
test_acc = [-999]*N_exps
loss = [-999]*N_exps
val_loss = [-999]*N_exps
test_loss = [-999]*N_exps
N_epochs = [-999]*N_exps

for idx, path_exp in enumerate(list_exp):
    acc[idx] = np.array (np.fromfile(os.path.join(path_exp, "accuracy.bin")))
    val_acc[idx] = np.array (np.fromfile(os.path.join(path_exp, "val_accuracy.bin")))
    loss[idx] = np.array (np.fromfile(os.path.join(path_exp, "loss.bin")))
    val_loss[idx] = np.array (np.fromfile(os.path.join(path_exp, "val_loss.bin")))

    N_epochs[idx] = loss[idx].shape[0]

    test_metrics = np.array (np.fromfile(os.path.join(path_exp,"test_metrics.bin")))
    test_loss[idx] = test_metrics[0]
    test_acc[idx] = test_metrics[1]

# prepare plots
setMatplotlibParam() 
#colors = cm.rainbow(np.linspace(0, 1, len(loss)))
colors = ["firebrick", "red", "darkcyan", "darkturquoise", "darkgoldenrod", "goldenrod"]

"""
# plot loss
fig = plt.figure(figsize=(5,3))
ax = fig.add_axes([0.05,0.05, 0.85, 0.85])
for i in range(N_exps):
    epochs_range = np.arange (0, N_epochs[i], 1)
    sns.lineplot    (x=epochs_range, y=loss[i], color = colors[i], label = os.path.basename(os.path.normpath(list_exp[0])), ax=ax)
    sns.lineplot    (x=epochs_range, y=val_loss[i], color = colors[i], label = os.path.basename(os.path.normpath(list_exp[0])),ax=ax)
    ax.scatter     (epochs_range[-1], test_loss[i], marker='o', s=20, color = colors[i])       
plt.savefig     (os.path.join(dir_all_models, 'loss.png'), bbox_inches= 'tight')
"""

# plot accuracy
fig = plt.figure(figsize=(5,3))
ax = fig.add_axes([0.05,0.05, 0.85, 0.85])
for i in range(N_exps):
    name_exp = os.path.basename(os.path.normpath(list_exp[i]))
    epochs_range = np.arange (0, N_epochs[i], 1)
    sns.lineplot    (x=epochs_range, y=acc[i], color = colors[2*i], label = name_exp + " + train", ax=ax)
    data_smooth = scipy.signal.savgol_filter(val_acc[i],11, 3)
    sns.lineplot    (x=epochs_range, y=data_smooth, color = colors[2*i+1], label = name_exp + " + valid", ax=ax)
    ax.scatter     (epochs_range[-1], test_acc[i], marker='o', s=20, color = colors[2*i+1])
ax.set_ylabel      ('Accuracy')
ax.set_xlabel      ('Epoch')
ax.legend      (loc=0, fontsize = 7)
plt.savefig     (os.path.join(dir_all_models, 'accuracy.png'), bbox_inches= 'tight')


#__________________________________________________________


