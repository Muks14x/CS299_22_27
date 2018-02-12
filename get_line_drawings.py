import numpy as np
import cv2
from glob import glob
import os
data = glob(os.path.join("imgs", "*.jpg"))

def get_image(image_path):
    return transform(cv2.imread(image_path, 1))

def transform(image, npx=512, is_crop=True):
    cropped_image = cv2.resize(image, (256, 256))
    return np.array(cropped_image)

base = np.array([get_image(sample_file) for sample_file in data])
base_edge = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in base])
base_edge = np.expand_dims(base_edge, 3)
output = np.tile(base_edge,3)

for i in range(len(data)):
    cv2.imwrite('out_' + data[i], output[i])

