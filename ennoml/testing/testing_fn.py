"""
 Author : Utkarsh
"""

import glob, io, os, shutil
from ennoml.testing.load_model import load_model
from ennoml.util.extra_fn import logs
from datetime import datetime
import pandas as pd
from ennoml.data.image_fn import load_image, get_img
import numpy as np


class model_fn():

    def __init__(self, img_types=None):
        if img_types is None:
            img_types = ['jpg', 'png', 'jpeg']
        self.size = 0
        self.labels = 0
        self.multi = False
        self.img_types = img_types
        self.crop_test = False
        self.pb = False
        self.print_full = False
        self.test_path = ""
        self.model_type = ""
        self.num_class = 0

    def get_model_file(self, path):
        def get_file(path, ext):
            return glob.glob(os.path.join(path, "*." + ext))

        for ext_ in ['hdf5', 'h5', 'pb']:
            models_list = get_file(path, ext_)
            if len(models_list) > 0:
                break
            else:
                continue
        if models_list[0].split(".")[-1] == 'pb':
            return path, get_file(path, 'config')[0]
        else:
            if len(models_list) == 1:
                return models_list[0], get_file(path, 'config')[0]
            elif len(models_list) > 1:
                return models_list[-1], get_file(path, 'config')[0]
            else:
                return path, None

    def read_config(self, file_path):
        if file_path is not None:
            data = eval(io.open(file_path, 'r').readlines()[0])
            return data['No_Of_Class'], data['Input_Size'], data['Model_Type'], data['Last_Layer'], data['Labels']
        else:
            return None

    def load_model_file(self, path,return_multi=False):
        if path.split(".")[-1] == "hdf5":
            model_file0 = path
            path = os.path.dirname(path)
            model_file, config = self.get_model_file(path)
            model_file = model_file0
        else:
            model_file, config = self.get_model_file(path)

        if config == None:
            logs(f'Model File Not Found - {model_file}',type='error',fn='print')
            logs(f'Or some file in the model folder are missing',type='error',fn='print')
            logs('Exiting the code.',type='error',fn='print')
            sys.exit()
        self.num_class, self.size, self.model_type, last_layer, self.labels = self.read_config(config)
        if model_file.split(".")[-1] == 'pb' or model_file == path:
            self.pb = True
        if type(self.labels[0]) == dict:
            self.multi = True
        logs(f'Loading Model - {model_file}',type='logs')
        model = load_model(self.num_class, self.size, model_file, self.model_type, last_layer).load()
        if return_multi:
            return model, self.labels, self.multi
        else:
            return model, self.labels


    def pred_flask(self,model,img_path,img_type='file',pos='no'):
        test_img = load_image(img_path, self.size,img_type=img_type, pos=pos)
        if self.pb:
            result = model(test_img)
        else:
            result = model.predict(test_img)
        if self.multi and self.model_type == "Multi_Type":
            result01 = [result[0][:self.num_class[0]], result[0][self.num_class[0]:]]
            result = result01
        pred_cls_, prob_ = custom_fn().get_class(result, self.labels, multi=self.multi)
        return pred_cls_,prob_


class custom_fn():

    def get_class(self, res, labels, multi=False):
        if multi:
            res0 = list(res[0].flatten())
            res1 = list(res[1].flatten())
            ind0 = res0.index(max(res0))
            ind1 = res1.index(max(res1))
            if labels[0][0] in ['Blur', 'Dark', 'Glare', 'Natural']:
                return [labels[1][ind1], labels[0][ind0]], [res1[ind1], res0[ind0]]
            else:
                return [labels[0][ind0], labels[1][ind1]], [res0[ind0], res1[ind1]]
        else:
            res = list(res.flatten())
            ind = res.index(max(res))
            return labels[ind], res[ind]