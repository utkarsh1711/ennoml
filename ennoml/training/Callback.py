"""
 Author : Utkarsh
"""

import tensorflow.keras.callbacks as CLB
import io,glob,os
from datetime import datetime
from tensorflow.python.training.saving import checkpoint_options as checkpoint_options_lib

class myCallback(CLB.Callback):

    def __init__(self, filepath,file_lim=4):
        self.filepath = filepath
        self._options = checkpoint_options_lib.CheckpointOptions()
        self.f_lim = file_lim
        self.sym_x = lambda path: "\\" if "\\" in path else "/"

    def on_epoch_end(self, epoch, logs={}):
        date0 = datetime.now()
        dateff0 = date0.strftime("%Y_%m_%d_%H")
        ff_path00 = self.filepath.split(".")
        if ff_path00[1] == "hdf5":
            filepath = ff_path00[0] + "_" + str(dateff0) + "." + ff_path00[1]
        else:
            filepath = self.filepath

        log_fff = io.open(ff_path00[0] + ".log", "r")
        log_data00 = log_fff.read()
        log_fff.close()

        logs_new_data00 = eval(log_data00)
        log_con_data = logs_new_data00['Logs']
        log11 = logs
        log11['epoch'] = epoch
        log_con_data.append(log11)

        logs_new_data00['Logs'] = log_con_data
        logs_new_data00['Save_Path'] = filepath
        # nn_dict_ = {}
        nn_dict_ = logs_new_data00['Epochs_Checkpoint']
        nn_dict_[ff_path00[0] + "_" + str(dateff0)] = epoch
        logs_new_data00['Epochs_Checkpoint'] = nn_dict_

        log_ffed = io.open(ff_path00[0] + ".log", "w")
        log_ffed.write(str(logs_new_data00))
        log_ffed.close()
        log_fff.close()
        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
        self.model.save(
            filepath, overwrite=True, options=self._options)
        if ff_path00[1] == "hdf5":
            sym_ = self.sym_x(filepath)
            new_path = self.filepath.split(sym_)
            new_path = sym_.join(new_path[:-1])
            file_list = glob.glob(new_path+f"{sym_}*.hdf5")
            file_list.sort()
            if len(file_list) > self.f_lim:
                while len(file_list) > self.f_lim:
                    if os.path.exists(file_list[0]):
                        os.remove(file_list[0])
                        file_list.remove(file_list[0])