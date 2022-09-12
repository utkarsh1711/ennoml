"""
 Author : Utkarsh
"""

from ennoml.util.extra_fn import gpu_fn, Params , logs

# Use any one of these functions to use GPU or CPU according to need
# gpu_fn().set_gpu_option("0", 0.60, memory=11.8)  # Define which GPU to use and How much percentage of it. # Define the % in value between 0 to# 1 -  (% / 100) the script will use
# gpu_fn().set_cpu_option()

import sys, os

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = "train.config"
var = Params(config_file)
print(config_file,var.Emptyval)
if var.Emptyval == 0:
    logs('Config File Not Found',type='error',fn='print')
    logs('Exiting the code.',type='error',fn='print')
    sys.exit()
settings = Params("main.settings")
if settings.Emptyval == 0:
    settings.root_dir = os.getcwd()
os.chdir(settings.root_dir)
if not var.gpu:
    gpu_fn().set_cpu_option()
    var.workers = 0

logs(f"[ROOT-DIR] {settings.root_dir}",type='logs',fn='print')

from ennoml.training import Loading_Model, Callback, train_fn
from ennoml.util import class_image
from datetime import datetime
import tensorflow.keras.callbacks as clb
import tensorflow as tf
from ennoml.testing.testing_fn import custom_fn
import pandas as pd

global date1
date0 = datetime.now()
date1 = date0.strftime("%Y_%m_%d")
date2 = date0.strftime("%Y/%m/%d-%H:%M:%S")


class train():

    def get_datagen(self, path, batch, size):
        datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.25, height_shift_range=0.25,
                                                                        rescale=1. / 255,
                                                                        rotation_range=360)  # , rescale=1. / 255 ,brightness_range=[0.4,0.8],
        datagen_out = datagenerator.flow_from_directory(directory=path, target_size=(size, size), batch_size=batch,
                                                        class_mode='categorical')
        return datagen_out

    def start_train(self, model, traingen, validgen, step_in_epoch, epochs, workers, lscallback, class_weight):
        if var.model_type in ['MobileNet', 'mobilenet', 'MobileNetV2', 'mobilenetv2',"MobileNetV3"]:
            if var.class_weiight:
                model.fit(traingen, steps_per_epoch=step_in_epoch, epochs=epochs,
                          workers=workers, callbacks=[lscallback, lscallback1], class_weight=class_weight)  # validation_data=validgen
            else:
                model.fit(traingen, steps_per_epoch=step_in_epoch, epochs=epochs,
                          workers=workers, callbacks=[lscallback, lscallback1])  # validation_data=validgen
        else:
            if var.class_weiight:
                if var.validpath is not None:
                    model.fit(traingen, steps_per_epoch=step_in_epoch, epochs=epochs,
                              workers=workers, callbacks=[lscallback],
                              class_weight=class_weight,validation_data=validgen)  # validation_data=validgen
                else:
                    model.fit(traingen, steps_per_epoch=step_in_epoch, epochs=epochs,
                          workers=workers, callbacks=[lscallback], class_weight=class_weight)  # validation_data=validgen
            else:
                model.fit(traingen, steps_per_epoch=step_in_epoch, epochs=epochs,
                          workers=workers, callbacks=[lscallback])  # validation_data=validgen


def training_log(data):
    logs_path, date_time = custom_fn().report(settings.root_dir,fn='Logs')
    log_file = os.path.join(logs_path, "Training_LOGS.xlsx")
    if not os.path.exists(log_file):
        df = pd.DataFrame([data],columns=['TimeStamp','ModelName','ModelType','Note'])
    else:
        df = pd.read_excel(log_file)
        df.loc[len(df.index)] = data
    df.to_excel(log_file,index=False)


def main():
    global lscallback1
    var.model_type = var.model_type.lower()
    if not var.multi_label and var.model_type in ['inception', 'mobilenet', 'mobilenetv2', 'mobilenetv3', 'shufflenet','effnet', 'inceptionv3']:
        last_layer = 'mixed9' # this value is only usable in the InceptionV3 model architecture.
        save_model_file, model_save_dir = train_fn.model_save(var.filepath, var.model_type)
        save_model_dir = save_model_file.split(".")[0]
        lscallback = Callback.myCallback(save_model_file)
        lscallback1 = clb.ModelCheckpoint(filepath=save_model_dir, verbose=1)
        traingen = train().get_datagen(var.trainpath, var.batchsize, var.size)
        validgen = var.validpath
        if var.validpath is not None:
            validgen = train().get_datagen(var.validpath, var.batchsize, var.size)
        labels_list = train_fn.create_labels(traingen.class_indices, var.filepath + "_" + date1, var.filepath,
                                             var.model_type)
        logs(f'Loading Model Achitecture - {var.model_type}',type='logs',fn='print')
        num_cls = len(labels_list)
        logs(labels_list,type='logs',fn='print')
        model = Loading_Model.single(num_cls, var.size, var.weight_file, var.retrain,
                                     loss_fn=tf.keras.losses.categorical_crossentropy,
                                     opti=tf.keras.optimizers.SGD(learning_rate=0.001), #Adam and adagrad
                                     metric=['categorical_accuracy'],
                                     activation='softmax',
                                     last_layer=last_layer)
        training_log([date2,model_save_dir,var.model_type,var.note])
        if var.model_type in ['inception', 'inceptionv3']:
            model = model.InceptionV3()
        elif var.model_type in ['mobilenet', 'mobilenetv2']:
            model = model.MobileNetV2(alpha=1)
        elif var.model_type in ['mobilenetv3']:
            model = model.MobileNetV3(alpha=1)
        elif var.model_type in ['ShuffleNet', 'shufflenet']:
            model = model.ShuffleNet()
        elif var.model_type in ['EffNet', 'effnet']:
            model = model.EffNet()
        train_path = var.trainpath

    if var.multi_label or var.model_type not in ['inception', 'mobilenet', 'mobilenetv2', 'mobilenetv3', 'shufflenet','effnet', 'inceptionv3']:
        logs('WRONG MODEL TYPE. Model Architecture not found in the code.',type='error',fn='print')
        logs('Exiting the code.',type='error',fn='print')
        sys.exit()

    class_weight = None
    if var.class_weiight:
        logs("Class_Weight_Balance - AUTO",type='logs',fn='print')
        class_weight = train_fn.get_class_weight(var.model_type, var.multi_label, train_path, labels_list)
        logs(class_weight,type='logs',fn='print')
    train_fn.create_config(labels_list, var.filepath, num_cls, var.size, var.weight_file, var.trainpath, var.model_type,
                           last_layer)
    logs(f"Model Save File - {model_save_dir}",type='logs',fn='print')
    class_image.main(var.trainpath, f"{model_save_dir}\\{var.filepath}", multi=var.multi_label)
    logs("Saving Class Names Image..",type='logs',fn='print')
    logs("Starting Training Process..",type='logs',fn='print')
    train().start_train(model, traingen, validgen, var.step_in_epoch, var.epochs, var.workers, lscallback, class_weight)
    logs(f"Training Completed",type='logs',fn='print')


if __name__ == "__main__":
    main()
