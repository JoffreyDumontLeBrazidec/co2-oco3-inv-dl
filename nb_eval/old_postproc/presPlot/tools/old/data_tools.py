#-------------------------------------------------
# dev/plumeDetection/models/postprocessing/
#--------------------------------------------------
# author  : joffreydumont@hotmail.fr
# created : 
#--------------------------------------------------
#
# Implementation of data_tools
# TODO: 
#
#

import sys
sys.path.append('/cerea_raid/users/dumontj/dev/plumeDetection/models/postprocessing/')
from local_importeur                    import 

from data.Data                          import Data
   
#------------------------------------------------------------------------
# import_data
def import_data(config, shuffleIndices):

    data                           = Data(config)
    data.prepareXCO2Data           (shuffleIndices = shuffleIndices)
    data.download_tt_dataset       ()
    return data

#------------------------------------------------------------------------
