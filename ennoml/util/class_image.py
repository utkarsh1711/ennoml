"""
 Author : Utkarsh
"""

import glob, os, io
from PIL import Image, ImageDraw, ImageFont

img_types = ['png', 'jpg', 'jpeg']
h1, w1 = 200, 200
h, w = 0, 0
br_ = 10
txt = 50
cols_ = 3
font_size = 12


def chk_lim(n,cols_):
    if n % cols_ == 0:
        return n // cols_
    else:
        return (n // cols_) + 1


def get_cords(cls_len,cols_):
    row_ = chk_lim(cls_len,cols_)
    cor_ = []
    y = br_
    for k in range(row_):
        x = br_
        for i in range(cols_):
            cor_.append([x, y])
            x = x + w1 + br_
        y = y + h1 + txt + br_
    return cor_


def img_list(file, multi=False, new_dir=""):
    if new_dir == "":
        if multi:
            n = file.split("\\")[-2]
            file = file + "\\Natural\\"
        else:
            n = file.split("\\")[-2]
    else:
        n = new_dir
    #         print(n)
    imglist = []
    for type_ in img_types:
        imglist0 = glob.glob(file + f"\\*.{type_}")
        if len(imglist0) > 0:
            imglist = imglist + imglist0
    if len(imglist) == 0:
        folder = glob.glob(file + "\\*\\")
        for i in folder:
            l1 = img_list(i,multi, n)[0]
            imglist = imglist + l1
    return imglist, n


def cols_no(cls_len):
    if 10 > cls_len > 3:
        return cls_len//2
    elif cls_len > 10:
        return 5
    else:
        return cols_


def main(f,save_name,multi=False):
    files = glob.glob(f + "\\*\\")
    cls_len = len(files)
    cols_ = cols_no(cls_len)
    if cls_len <= cols_:
        h, w = (h1 + (br_ * 2)) + 50, ((w1 * cols_) + (br_ * 4))
    else:
        row_ = chk_lim(cls_len,cols_)
        h, w = ((h1 + txt + br_) * row_), ((w1 * cols_) + (br_ * (cols_+1)))

    image = Image.new("RGB", (w, h), color='white')
    for ind, cls_ in enumerate(files):
        print(ind+1,"/",len(files),end="\r")
        img_name, cls_name = img_list(cls_,multi=multi)
        cords_ = get_cords(cls_len,cols_)
        font = ImageFont.truetype("arial.ttf", font_size)
        #     print(img_name,cls_name)
        if len(img_name)<1:
            im0_ = Image.new("RGB", (w1, h1), color='black')
        else:
            img_name = img_name[0]
            im0_ = Image.open(img_name)
            im0_ = im0_.resize((w1, h1))
        image.paste(im0_, (cords_[ind][0], cords_[ind][1]))
        d1 = ImageDraw.Draw(image)
        d1.text((5 + cords_[ind][0], 5 + cords_[ind][1] + h1), cls_name, fill="black", font=font)
        image.save(f"{save_name}.jpg")
