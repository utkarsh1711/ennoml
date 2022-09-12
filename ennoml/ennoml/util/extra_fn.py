"""
 Author : Utkarsh
"""

import tensorflow as tf
import os
from datetime import datetime


class Params(object):
    def __init__(self, input_file_name):
        try:
            with open(input_file_name, 'r') as input_file:
                file = input_file.read()
                file_var_ = eval(file)
                for key_ in file_var_.keys():
                    attr = key_
                    value = (file_var_[key_])
                    self.__dict__[attr] = value
                self.__dict__['Emptyval'] = 1
        except:
            self.__dict__['Emptyval'] = 0



class gpu_fn():

    def __init__(self):
        logs("[GPU] FN- Called",type='logs',fn='print')

    def set_gpu_option(self,which_gpu, fraction_memory, memory=12):
        which_gpu = int(which_gpu)
        size_in_mb = 1024 * int(memory * fraction_memory)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[which_gpu], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[which_gpu], True)
        tf.config.experimental.set_virtual_device_configuration(gpus[which_gpu], [
            tf.config.LogicalDeviceConfiguration(memory_limit=size_in_mb)])
        logs(f"[GPU] Using - {fraction_memory*100}% of {gpus[which_gpu]}",type='logs',fn='print')

    def set_cpu_option(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logs("[CPU] Only CPU  is using",type='logs',fn='print')



def logs(data,type="logs",fn='print'):
    date0 = datetime.now()
    datenow = date0.strftime("%Y/%m/%d-%H:%M:%S")
    if type == 'logs':
        if fn == 'print':
            print(f'[ENNOML][INFO][{datenow}] : {data}')
    elif type == 'error':
        if fn == 'print':
            print(f'[ENNOML][ERROR<!>][{datenow}] : {data}')