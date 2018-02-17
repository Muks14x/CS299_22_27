# -*- coding: utf-8 -*-

import numpy as np
import cv2
from glob import glob
import lmdb

#def transform(image, size=256):
#    cropped_image = cv2.resize(image, (size, size))
#    return np.array(cropped_image)

output_images = glob('/home/muks/Innolab/InnoLab/imgs/*.jpg')
input_images = glob('/home/muks/Innolab/InnoLab/out_imgs/*.jpg')

img = cv2.imread('/home/muks/Innolab/InnoLab/imgs/1.jpg', 1)

import caffe.proto.caffe_pb2 as caffe_pb2

INP_SIZE = 256

def make_datum(img):
    return caffe_pb2.Datum(
        channels=3,
        width=INP_SIZE,
        height=INP_SIZE,
        data=np.rollaxis(img, 2).tostring())

train_lmdb_color = '/home/muks/Innolab/InnoLab/train_lmdb_color'
train_lmdb_line = '/home/muks/Innolab/InnoLab/train_lmdb_line'

in_db = lmdb.open(train_lmdb_color, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(output_images):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        datum = make_datum(img)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
in_db.close()

in_db = lmdb.open(train_lmdb_line, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(input_images):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        datum = make_datum(img)
        in_txn.put('{:0>5d}'.format(in_idx).encode(), datum.SerializeToString())
in_db.close()
