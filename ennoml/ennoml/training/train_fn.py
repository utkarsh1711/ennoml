"""
 Author : Utkarsh
"""

from datetime import datetime
import io, os, glob
import pandas as pd

global date
global date1
date0 = datetime.now()
date = date0.strftime("%Y_%m_%d_%H")
date1 = date0.strftime("%Y_%m_%d")


def model_save(filepath,model_type):
    save_folder = os.path.join("Models",model_type, filepath + "_" + str(date))   # only Model folder path
    save_file = os.path.join(save_folder, filepath + ".hdf5")  # Model path with  model file name
    os.makedirs(save_folder, exist_ok=True)
    return save_file, save_folder        

def create_config(labels_list, filepath, num_cls, size, old_model, trainpath, model_type, last_layer):
    config = {}
    log = {}
    config['Name'] = filepath+ "_" +date
    config['Model_Type'] = model_type
    config['Last_Layer'] = last_layer
    config['No_Of_Class'] = num_cls
    config['Labels'] = labels_list
    config['Input_Size'] = size
    config['Weight_File'] = old_model
    log['Save_Path'] = ""
    config['Train_Data'] = img_count(trainpath)
    config['Train_Path'] = trainpath
    log['Logs'] = []
    log['Epochs_Checkpoint'] = {}
    get_file_dir = model_save(filepath,model_type)[0].split(".")[0]
    file = io.open(get_file_dir + ".config","w+")
    file.write(str(config))
    file.close()
    with io.open(get_file_dir + ".log","w+") as file:
        file.write(str(log))

def create_labels(cls_ind , model_name, filepath,model_type,get=True):
    save_dir = model_save(filepath,model_type)[1]
    label = [""] * len(cls_ind)
    for i, j in enumerate(cls_ind):
        label[i] = j
    file_lab = io.open(save_dir +"\\Label_" + model_name.split(".")[0] + "_.txt", "w")
    file_lab.write("\n".join(label))
    file_lab.close()
    if get:
        return label


def img_count(path):
    file = {}
    list = []
    total = 0
    dir = glob.glob(path + "/*")
    for i in dir:
        name = i.split("\\")[-1]
        count = sum(len(files) for _, _, files in os.walk(i))
        file[name] = count
        list.append([name, count])
        total = total + count
    list.append(['Total',total])
    return list


def get_class_weight(model_type, multi, train_path,labels):
    if model_type in ["Multi_Type"] and not multi:
        df = pd.read_excel(train_path)
        cols_ = df.columns[1:]
        vals_ = []
        v_weight = {}
        for i in cols_:
            vals_.append(df[i].sum())
        m = max(vals_)
        for i in vals_:
            v_weight[vals_.index(i)] = round((m / i), 1)
        return v_weight
    elif not multi and model_type in ['Inception', 'inception', 'InceptionV3', 'inceptionv3','MobileNet', 'mobilenet', 'MobileNetV2', 'mobilenetv2','MobileNetV3']:
        t_dict = {}
        t_count = img_count(train_path)
        vals_ = []
        for i in t_count:
            t_dict[i[0]] = i[1]
        for i in labels:
            vals_.append(t_dict[i])
        v_weight = {}
        m = max(vals_)
        # print(vals_)
        for ind,i in enumerate(vals_):
            v_weight[ind] = round((m / i), 1)
        return v_weight
    elif multi:
        def get_cw(vals_):
            v_weight = {}
            m = max(vals_)
            for i in vals_:
                v_weight[vals_.index(i)] = round((m / i), 2)
            return v_weight
        df = pd.read_excel(train_path)
        cols_ = df.columns[1:]
        l1 = len(labels[0].keys())
        vals_ = []
        for i in cols_:
            vals_.append(df[i].sum())
        val_0 = vals_[:l1]
        val_1 = vals_[l1:]
        c_w0 = get_cw(val_0)
        c_w1 = get_cw(val_1)
        # return {'output0': c_w0, 'output1': c_w1}
        return get_cw(vals_)
    else:
        t_dict = {}
        t_count = img_count(train_path)
        vals_ = []
        for i in t_count:
            t_dict[i[0]] = i[1]
        for i in labels:
            vals_.append(t_dict[i])
        v_weight = {}
        m = max(vals_)
        for i in vals_:
            v_weight[vals_.index(i)] = round((m / i), 1)
        return v_weight