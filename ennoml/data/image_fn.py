"""
 Author : Utkarsh
"""

import io,os,glob
from PIL import Image
from ennoml.data.aug_fn import image_aug
import numpy as np


def get_img(file,img_types, new_dir="",return_filename=False):
    n = ""
    if new_dir == "":
        n = file.split("\\")[-2]
    imglist = []
    for type_ in img_types:
        imglist0 = glob.glob(file + f"*.{type_}")
        if len(imglist0) > 0:
            imglist = imglist + imglist0
    folder = glob.glob(file + "*\\")
    for i in folder:
        l1 = get_img(i,img_types, n)
        imglist = imglist + l1
    if return_filename:
        return imglist, n
    else:
        return imglist


def load_pil(img_,size):
    w, h = img_.size
    if w != size or h != size:
        img_New = image_aug().resize_(img_, size)
    else:
        img_New = img_
    img_array = np.asarray(img_New)
    img_0 = np.true_divide(img_array, 255)
    img_1 = np.expand_dims(img_0, axis=0)
    return img_1


def load_image(path, size,img_type='file', pos='no', z=0.6):
    if img_type=='file':
        img_ = Image.open(path)
    elif img_type=='array':
        img_ = Image.fromarray(path)  # If using cv2 array first convert from BGR to RGB.
    if pos == 'no':
        return load_pil(img_,size)
    if pos == 'top':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.15)
        top = 0
        right = int(w * 0.85)
        bottom = int(h * 0.5)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_top.jpg")
        return load_pil(img_,size)
    if pos == 'bottom':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.15)
        top = int(h * 0.5)
        right = int(w * 0.85)
        bottom = int(h)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_bottom.jpg")
        return load_pil(img_,size)
    if pos == 'centre':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.25)
        top = int(h * 0.25)
        right = int(w * 0.75)
        bottom = int(h * 0.75)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_centre.jpg")
        return load_pil(img_,size)
    if pos == 'centre_top':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.25)
        top = int(h * 0.125)
        right = int(w * 0.75)
        bottom = int(h * 0.625)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_centre_top.jpg")
        return load_pil(img_,size)
    if pos == 'centre_bottom':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.25)
        top = int(h * 0.375)
        right = int(w * 0.75)
        bottom = int(h * 0.875)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_centre_bottom.jpg")
        return load_pil(img_,size)
    if pos == 'centre_left':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.125)
        top = int(h * 0.25)
        right = int(w * 0.625)
        bottom = int(h * 0.75)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_centre_left.jpg")
        return load_pil(img_,size)
    if pos == 'centre_right':
        img_ = image_aug().resize_(img_, size * 3)
        w, h = img_.size
        left = int(w * 0.375)
        top = int(h * 0.25)
        right = int(w * 0.875)
        bottom = int(h * 0.75)
        img_ = img_.crop((left, top, right, bottom))
        # img_.save(f"{crp_p}\\{im_name.split('.')[0]}_centre_right.jpg")
        return load_pil(img_,size)
    if pos == 'zoom_out':
        img_ = image_aug().resize_(img_, size, z)
        return load_pil(img_,size)
