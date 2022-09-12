"""
 Author : Utkarsh
"""
import os,io
from ennoml.util.extra_fn import gpu_fn,Params,logs
import cv2

config_file = "ennoml/test.config"  # Path of the config file
var = Params(config_file)

if var.Emptyval == 0:
    logs('Config File Not Found OR File Format is WRONG!!',type='error')
    logs('Exiting the code.',type='error')
    sys.exit()

if not var.gpu:
    gpu_fn().set_cpu_option()

from ennoml.testing.testing_fn import model_fn

model_main_fn = model_fn()
model, labels = model_main_fn.load_model_file(var.model_path)
logs(f'Model Loaded.',type='logs')

test_img_path = "ennoml/test1.jpeg"  # path of the image file
path = test_img_path
logs(f'Loading image from file',type='logs')
output, prob = model_main_fn.pred_flask(model, path,img_type='file', pos="no")
logs(f'Pred Class - {output}, Probability - {prob}',type='logs')


img = cv2.imread(test_img_path)
im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # If using CV2 for image convert the image to RGB before sending.
logs(f'Loading image from array',type='logs')
output, prob = model_main_fn.pred_flask(model, im_rgb,img_type='array', pos="no")
logs(f'Pred Class - {output}, Probability - {prob}',type='logs')