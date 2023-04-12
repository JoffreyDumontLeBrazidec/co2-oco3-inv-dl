# -------------------------------------------------
# dev/plumeDetection/models/postprocessing/
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of pres_cm
# TODO:
#
#

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

from pres_config import *

df_test = pd.read_csv(
    os.path.join(dir_save_current_model, "list_infos_predictions.csv"), index_col=0
)

binary_test_predictions = np.around(df_test.loc[:, "pred_pos_perc"].to_numpy())

cm = metrics.confusion_matrix(df_test.loc[:, "positivity"], binary_test_predictions, labels=np.array([0, 1]))

disp = metrics.ConfusionMatrixDisplay(cm)
disp.plot()
plt.savefig(
    os.path.join(dir_save_current_model, "confusion_matrix.png"), bbox_inches="tight"
)

