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

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from chain_config                import *
from local_importeur            import *


# download dataframe
df        = pd.read_csv (dir_all_models + "/" + "table_combinations.csv")
N_configs = df.shape[0]
first_folder = int(0)
last_folder = int(N_configs)
N_folders = last_folder - first_folder

# fill dataframe
## loss
list_train_loss = np.ones ((N_folders)) * (-999)
list_valid_loss = np.ones ((N_folders)) * (-999)
list_test_loss = np.ones ((N_folders)) * (-999)
list_N_epochs = np.ones ((N_folders)) * (-999)
for index_config in range (first_folder, last_folder):
    dir_save_current_model  = dir_all_models + "/test%s/"%index_config
    if (os.path.exists(dir_save_current_model + "/loss.bin")): # numpy array "proxy" appearing only when test completed
        list_train_loss[index_config]  = np.min (np.array (np.fromfile(dir_save_current_model + "/" + "loss.bin")))
        list_valid_loss[index_config]  = np.min (np.array (np.fromfile(dir_save_current_model + "/" + "val_loss.bin")))
        list_test_loss[index_config]   = np.array (np.fromfile(dir_save_current_model + "/" + "test_metrics.bin"))[0]
        list_N_epochs[index_config]    = len(np.array (np.fromfile(dir_save_current_model + "/" + "loss.bin"))) 

df["train_loss.best"] = list_train_loss
df["valid_loss.best"] = list_valid_loss
df["test_loss.best"]  = list_test_loss
df["N_epochs"] = list_N_epochs.astype(int)

## accuracy
list_train_accuracy = np.ones ((N_folders)) * (-999)
list_valid_accuracy = np.ones ((N_folders)) * (-999)
list_test_accuracy = np.ones ((N_folders)) * (-999)
for index_config in range (first_folder, last_folder):
    dir_save_current_model  = dir_all_models + "/test%s/"%index_config
    if (os.path.exists(dir_save_current_model + "/accuracy.bin")): # numpy array "proxy" appearing only when test completed
        list_train_accuracy[index_config]   = np.max (np.array (np.fromfile(dir_save_current_model + "/" + "accuracy.bin")))
        list_valid_accuracy[index_config]   = np.max (np.array (np.fromfile(dir_save_current_model + "/" + "val_accuracy.bin")))   
        list_test_accuracy[index_config]    = np.array (np.fromfile(dir_save_current_model + "/" + "test_metrics.bin"))[1]

if np.max(list_train_accuracy)>-999: 
    df["train_accuracy.best"] = list_train_accuracy
    df["valid_accuracy.best"] = list_valid_accuracy
    df["test_accuracy.best"] = list_test_accuracy


# clean dataframe
df = df.drop (df[df["N_epochs"] == -999].index)

## wind_as_input
wind_as_input = [None] * df.shape[0]
for index_test, norm_index_test in zip(df.index.values, range (len(df.index.values))):
    wind_as_input [norm_index_test] = df.at[index_test, "winds.format"]
    if wind_as_input [norm_index_test] == "fields":
        wind_as_input [norm_index_test] = "field" + "[" + str(df.at[index_test, "N_wind_fields"]) + "ch]"
df["wind_as_input"] = wind_as_input 
    
## dynamics_as_input
dynamic_as_input = [None] * df.shape[0]
for index_test, norm_index_test in zip(df.index.values, range (len(df.index.values))):
    dynamic_as_input [norm_index_test] = df.at[index_test, "dynamic.format"]
    if dynamic_as_input [norm_index_test] == "fields":
        dynamic_as_input [norm_index_test] = "field" + "[" + str(df.at[index_test, "N_dynamic_fields"]) + "ch]"
df["dynamic_as_input"] = dynamic_as_input   
    
# save
df.to_csv (dir_all_models + "/" + "filled_table_combinations.csv")
