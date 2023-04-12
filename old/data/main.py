# -------------------------------------------------
# dev/plumeDetection/main
# --------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created :
# --------------------------------------------------
#
# Implementation of plume-detection
# TODO:
#
#

from importeur import *
from Data import Data

if __name__ == "__main__":

    dir_save_current_model = "/cerea_raid/users/dumontj/dev/plumeDetection/models/model_weights/chaintest2/test3/"

    config = TreeConfigParser(comment_char="//")
    configFile = dir_save_current_model + "/" + "config.cfg"
    config.readfiles(configFile)

    data_1 = Data(config)
    data_2 = Data(config)
    shuffleIndices = np.array(
        np.fromfile(dir_save_current_model + "/" + "shuffle_indices.bin")
    ).astype("int")
    data_1.prepareXCO2Data()
    data_2.prepareXCO2Data(shuffleIndices)

    print(
        "train",
        len(data_1.y_train),
        "test",
        len(data_1.y_test),
        "valid",
        len(data_1.y_validation),
    )
    print(
        "train",
        len(data_2.y_train),
        "test",
        len(data_2.y_test),
        "valid",
        len(data_2.y_validation),
    )

    print(
        "train",
        np.sum(data_1.y_train),
        "test",
        np.sum(data_1.y_test),
        "valid",
        np.sum(data_1.y_validation),
    )
    print(
        "train",
        np.sum(data_2.y_train),
        "test",
        np.sum(data_2.y_test),
        "valid",
        np.sum(data_2.y_validation),
    )

    print(
        "train",
        np.sum(data_1.y_train) / len(data_1.y_train),
        "test",
        np.sum(data_1.y_test) / len(data_1.y_test),
        "valid",
        np.sum(data_1.y_validation) / len(data_1.y_validation),
    )
    print(
        "train",
        np.sum(data_2.y_train) / len(data_2.y_train),
        "test",
        np.sum(data_2.y_test) / len(data_2.y_test),
        "valid",
        np.sum(data_2.y_validation) / len(data_2.y_validation),
    )
