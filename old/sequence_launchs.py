----------------------------------------------------------------------------
# author: joffrey.dumont@enpc.fr or joffreydumont@hotmail.fr
# created: 2021/2022 
# laboratory: CEREA,  École des Ponts and EDF R&D, Île-de-France, France
# project: Prototype system for a Copernicus C02 monitoring service (CoCO2)
#----------------------------------------------------------------------------

import bash
import os
import sys
import glob
import psutil

if __name__ == "__main__":
    
    in_command = bash.config_file_names_from_command()
    program = "main.py"
    base_config_file = "config.cfg"
    path = in_command[0]

    if len(in_command) == 1:
        first_folder = 0
        last_folder = len(
            [subdir for subdir in glob.glob(os.path.join(path, "config*")) if os.path.isdir(subdir)]
        )

    elif len(in_command) == 3:
        first_folder = int(in_command[1])
        last_folder = int(in_command[2])

    for i in range(first_folder, last_folder):

        directory_config = os.path.join(path, "config%s"%i)
        current_config_file = os.path.join(directory_config, base_config_file)
        if os.path.exists(current_config_file):
            print("\n \n \n \n")
            print("Launcher number ", i)            
            print (current_config_file)
            print("\n")
            command = " ".join(["ipython", program, current_config_file])
            print(command)
            os.system(command)
        else:
            print("Config. file ", configfile, "DOES NOT exists")
