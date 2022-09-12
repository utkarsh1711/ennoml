"""
 Author : Utkarsh
"""

from PIL import Image, ImageOps, ImageEnhance
import random
import cv2
import numpy as np
from skimage.util import random_noise


# import os


class image_aug():

    def rotate_(self, img_, angle=[20, 360]):
        if type(angle) == list:
            range_ = random.randint(angle[0], angle[1])
            return img_.rotate(range_)
        elif type(angle) == int:
            return img_.rotate(angle)
        else:
            return None

    def shift_(self, img_, value=[0.1, 0.2], type_=['L', "R", 'T', 'B']):
        px_min = int(img_.size[0] * value[0])
        px_max = int(img_.size[0] * value[1])
        shift_px = random.randint(px_min, px_max)
        if 'L' in type_:
            img_left = Image.new("RGB", img_.size, color='black')
            img_left.paste(img_, (-shift_px, 0))
            return img_left
        if 'R' in type_:
            img_right = Image.new("RGB", img_.size, color='black')
            img_right.paste(img_, (shift_px, 0))
            return img_right
        if 'T' in type_:
            img_top = Image.new("RGB", img_.size, color='black')
            img_top.paste(img_, (0, -shift_px))
            return img_top
        if 'B' in type_:
            img_bottom = Image.new("RGB", img_.size, color='black')
            img_bottom.paste(img_, (0, shift_px))
            return img_bottom

    def resize_(self, img_, size, type_='PIL', z=0):
        if type_ == 'PIL':
            if z != 0:
                z_size = int(size * z)
            else:
                z_size = size
            scale = min(z_size / img_.size[0], z_size / img_.size[1])
            nw = int(scale * img_.size[0])
            nh = int(scale * img_.size[1])
            # print(nw,nh,img_.size,scale,z_size)
            image1 = img_.resize((nw, nh))
            if nw < size or nh < size:
                top, left = int((size - nh) / 2), int((size - nw) / 2)
                image = Image.new("RGB", (size, size), color='black')
                image.paste(image1, (left, top))
                exif = img_.getexif()
                for k in exif.keys():
                    if k != 0x0112:
                        exif[k] = None  # If I don't set it to None first (or print it) the del fails.
                        del exif[k]
                new_exif = exif.tobytes()
                image.info["exif"] = new_exif
                image = ImageOps.exif_transpose(image)
                return image
            else:
                exif = img_.getexif()
                for k in exif.keys():
                    if k != 0x0112:
                        exif[k] = None  # If I don't set it to None first (or print it) the del fails.
                        del exif[k]
                new_exif = exif.tobytes()
                image1.info["exif"] = new_exif
                image1 = ImageOps.exif_transpose(image1)
                return image1
        if type_ == 'cv2':
            if z != 0:
                z_size = int(size * z)
            else:
                z_size = size
            scale = min(z_size / img_.shape[0], z_size / img_.shape[1])
            nw = int(scale * img_.shape[1])
            nh = int(scale * img_.shape[0])
            image1 = cv2.resize(img_, (nw, nh))
            if nw < size or nh < size:
                top, left = int((size - nw) / 2), int((size - nh) / 2)
                image = np.zeros((size, size, 3), np.uint8)
                image[left:left + nh, top:top + nw] = image1
                return image
            else:
                return image1

    def img_crop_(self, img_, width_px, height_px):
        w, h = img_.size
        px_w = w * width_px
        px_h = h * height_px
        top = px_h
        left = px_w
        right = w - px_w
        bottom = h - px_h
        im1 = img_.crop((left, top, right, bottom))
        return im1

    def img_brightness(self, img_, range=[1.4, 1.7]):
        enhancer = ImageEnhance.Brightness(img_)
        xmin, xmax = range[0], range[1]
        factor = round(random.uniform(xmin, xmax), 2)
        im_output = enhancer.enhance(factor)
        return im_output

    def img_contrast(self, img_, range=[1.4, 1.8]):
        enhancer = ImageEnhance.Sharpness(img_)
        xmin, xmax = range[0], range[1]
        factor = round(random.uniform(xmin, xmax), 2)
        im_output = enhancer.enhance(factor)
        return im_output

    def noise_(self, img_, value=0.5):
        if type(value) == float:
            value = value
        elif type(value) == list:
            xmin, xmax = value[0], value[1]
            value = round(random.uniform(xmin, xmax), 2)
        img_array = np.asarray(img_)
        noise = random_noise(img_array, mode='s&p', amount=value)
        noise = np.array(255 * noise, dtype=np.uint8)
        image = Image.fromarray(noise)
        return image

